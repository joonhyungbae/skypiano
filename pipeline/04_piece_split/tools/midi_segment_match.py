"""
Segment-level MIDI piece identification.

Pipeline:
  1. Split session MIDI by silence (midi_segment.segment_midi).
  2. Fingerprint each segment as a whole.
  3. For each segment, accumulate hash hits against the anchor DB and score
     each anchor take by (matched_fps / min(seg_fps, anchor_fps)).
  4. Emit the top-K anchor candidates per segment.

Why this beats whole-session matching:
  A segment is ≈1 piece, so fingerprint coverage is dense and the winning
  anchor ratio becomes meaningful. Random hash collisions get diluted because
  they don't cluster temporally within the segment.
"""
from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass, asdict
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # pipeline/04_piece_split/
from src.midi_fingerprint import (  # noqa: E402
    AnchorMeta, AnchorRef,
    build_db, events_from_notes, fingerprint,
)
from midi_segment import Segment, segment_midi  # noqa: E402


@dataclass
class SegmentMatch:
    seg_index: int
    seg_start: float
    seg_end: float
    seg_duration: float
    seg_n_events: int
    seg_n_fingerprints: int
    candidates: list[dict]   # top-K [{piece_id, take_id, score, matched, anchor_fps, offset_consistency}]


def _offset_consistency(offsets: list[float], bin_sec: float = 0.5) -> tuple[int, float]:
    """Return (mode_bin_count, mode_bin_center) — how many hits share the dominant offset bin."""
    if not offsets:
        return 0, 0.0
    buckets: dict[int, int] = defaultdict(int)
    for o in offsets:
        buckets[int(o / bin_sec)] += 1
    best_bin, best_cnt = max(buckets.items(), key=lambda kv: kv[1])
    return best_cnt, (best_bin + 0.5) * bin_sec


def match_segment(seg: Segment,
                  db: dict[int, list[AnchorRef]],
                  meta: dict[tuple[str, str], AnchorMeta],
                  *,
                  top_k: int = 5,
                  min_matched: int = 3,
                  ) -> SegmentMatch:
    events = events_from_notes(seg.notes)
    fps = fingerprint(events)

    # (piece_id, take_id) → list of (anchor_time - query_rel_time) offsets
    hits: dict[tuple[str, str], list[float]] = defaultdict(list)
    for h, q_rel_t in fps:
        if h not in db:
            continue
        for ref in db[h]:
            hits[(ref.piece_id, ref.take_id)].append(q_rel_t - ref.anchor_time)

    scored: list[dict] = []
    for (pid, tid), offs in hits.items():
        m = meta.get((pid, tid))
        if m is None or m.n_fingerprints == 0:
            continue
        matched = len(offs)
        if matched < min_matched:
            continue
        mode_cnt, mode_off = _offset_consistency(offs)
        denom = min(len(fps), m.n_fingerprints) or 1
        score = mode_cnt / denom
        scored.append({
            "piece_id": pid,
            "take_id": tid,
            "score": round(score, 3),
            "matched": matched,
            "coherent": mode_cnt,
            "anchor_fps": m.n_fingerprints,
            "offset": round(mode_off, 2),
        })

    scored.sort(key=lambda d: -d["score"])
    return SegmentMatch(
        seg_index=-1,
        seg_start=seg.start,
        seg_end=seg.end,
        seg_duration=seg.duration,
        seg_n_events=len(events),
        seg_n_fingerprints=len(fps),
        candidates=scored[:top_k],
    )


def match_session(midi_path: Path | str,
                  db: dict[int, list[AnchorRef]],
                  meta: dict[tuple[str, str], AnchorMeta],
                  *,
                  gap_threshold: float = 3.0,
                  top_k: int = 5,
                  ) -> list[SegmentMatch]:
    segments = segment_midi(midi_path, gap_threshold=gap_threshold)
    out: list[SegmentMatch] = []
    for i, seg in enumerate(segments):
        sm = match_segment(seg, db, meta, top_k=top_k)
        sm.seg_index = i
        out.append(sm)
    return out


def _fmt(sm: SegmentMatch) -> str:
    head = f"[{sm.seg_index:>3}] t={sm.seg_start:>7.1f}-{sm.seg_end:>7.1f} " \
           f"dur={sm.seg_duration:>5.1f}s fps={sm.seg_n_fingerprints:>4}"
    if not sm.candidates:
        return f"{head}  (no match)"
    top = sm.candidates[0]
    tail = f"  → {top['piece_id']!r:30s} score={top['score']:.2f} " \
           f"({top['coherent']}/{top['anchor_fps']}) take={top['take_id']}"
    if len(sm.candidates) > 1:
        nxt = sm.candidates[1]
        tail += f"  |next: {nxt['piece_id']!r} {nxt['score']:.2f}"
    return head + tail


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("usage: python midi_segment_match.py <midi> [--gap 3.0] [--json out.json]")
        sys.exit(0)
    mid = sys.argv[1]
    gap = 3.0
    out_json = None
    for i, a in enumerate(sys.argv):
        if a == "--gap" and i + 1 < len(sys.argv):
            gap = float(sys.argv[i + 1])
        if a == "--json" and i + 1 < len(sys.argv):
            out_json = sys.argv[i + 1]

    db, meta = build_db()
    print()
    results = match_session(mid, db, meta, gap_threshold=gap)
    for sm in results:
        print(_fmt(sm))
    n_matched = sum(1 for sm in results if sm.candidates)
    print(f"\n{n_matched}/{len(results)} segments matched a piece")

    if out_json:
        Path(out_json).parent.mkdir(parents=True, exist_ok=True)
        with open(out_json, "w", encoding="utf-8") as f:
            json.dump([asdict(sm) for sm in results], f, ensure_ascii=False, indent=2)
        print(f"Saved: {out_json}")
