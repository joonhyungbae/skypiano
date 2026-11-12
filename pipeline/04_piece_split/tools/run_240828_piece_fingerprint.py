"""
Per-piece MIDI fingerprint matching for 240828 — using 240828-only anchors.

Anchor source: Data/Anchors/2024/ (built by build_240828_anchors.py from
reviewer-confirmed Phase 1/2 labels). This is intentional:
  - 220819 official anchors have signal/length mismatch with 240828 queries
    (different MIDI keyboard capture session, often very short snippets like 2-7s).
  - 2024 anchors are full-length performances on the same instrument and
    room — symbolic fingerprint scores are cleaner and more discriminative.

Each query is normalized to t=0 and matched against same-task anchors. The
piece's own take is excluded so confirmed pieces don't trivially self-match.

Output: Data/Metadata/240828/piece_fingerprint_matches.json
"""
from __future__ import annotations

import functools
import json
import re
import sys
from collections import defaultdict
from dataclasses import dataclass, replace
from pathlib import Path

_HERE = Path(__file__).resolve()
sys.path.insert(0, str(_HERE.parent.parent))
from src.midi_fingerprint import (  # noqa: E402
    AnchorRef, Event, extract_events, fingerprint,
)

print = functools.partial(print, flush=True)

PIECE_ROOT = Path("Data/Pipeline/04_piece_split/240828")
OUT_PATH = Path("Data/Metadata/240828/piece_fingerprint_matches.json")
ANCHOR_ROOT = Path("Data/Anchors/2024")
TASK_DIRS = {"Technique1", "Technique2", "Set1", "Free"}

MIN_RATIO = 0.10
MIN_VOTES_FLOOR = 2
OFFSET_BIN_SEC = 0.5
TOP_K = 5

# Stage 04 task tag → 2024 anchor task directory.
TASK_TO_ANCHOR = {0: "Technique1", 1: "Technique2", 2: "Set1", 9: "Free"}

PIECE_FILE_RE = re.compile(r"^(240828_C\d+_N\d+_T(\d+)_P\d+_t\d+)_p(\d+)\.mid$")


@dataclass
class TakeMeta:
    n_fingerprints: int
    duration_sec: float
    task: str


def _normalize(events: list[Event]) -> list[Event]:
    """Shift events so the first onset is at t=0."""
    if not events:
        return []
    t0 = events[0].onset
    if t0 == 0.0:
        return events
    return [replace(e, onset=e.onset - t0) for e in events]


def build_normalized_db():
    """Build hash → [AnchorRef] DB from 2024 anchors only, normalized to t=0.

    Empty if Data/Anchors/2024/ doesn't exist yet (run build_240828_anchors.py
    first).
    """
    db: dict[int, list[AnchorRef]] = defaultdict(list)
    meta: dict[tuple[str, str], TakeMeta] = {}
    n_takes = 0
    if not ANCHOR_ROOT.is_dir():
        print(f"[ERR] {ANCHOR_ROOT} not found. Run build_240828_anchors.py first.")
        return db, meta

    for task_dir in ANCHOR_ROOT.iterdir():
        if not task_dir.is_dir() or task_dir.name not in TASK_DIRS:
            continue
        for piece_dir in task_dir.iterdir():
            if not piece_dir.is_dir():
                continue
            piece_id = piece_dir.name
            for mid in piece_dir.glob("*.mid"):
                take_id = mid.stem
                events = _normalize(extract_events(mid))
                fps = fingerprint(events)
                for h, t in fps:
                    db[h].append(AnchorRef(piece_id, take_id, t))
                meta[(piece_id, take_id)] = TakeMeta(
                    n_fingerprints=len(fps),
                    duration_sec=events[-1].onset - events[0].onset if events else 0.0,
                    task=task_dir.name,
                )
                n_takes += 1
    avg_refs = sum(len(v) for v in db.values()) / max(len(db), 1)
    print(f"[anchor-db 2024] {n_takes} takes → {len(db)} unique hashes "
          f"(avg {avg_refs:.1f} refs/hash)")
    return db, meta


def match_piece(query_path: Path, db, meta, anchor_task: str, exclude_take: str):
    """Match a normalized piece MIDI against the 2024 anchor DB.

    Filters to anchors of the requested task; excludes the query piece's own
    take so confirmed pieces don't trivially self-match.
    """
    query_events = _normalize(extract_events(query_path))
    query_fps = fingerprint(query_events)

    votes: dict[tuple[str, str, int], int] = defaultdict(int)
    sums: dict[tuple[str, str, int], float] = defaultdict(float)

    for q_hash, q_time in query_fps:
        if q_hash not in db:
            continue
        for ref in db[q_hash]:
            tk = meta.get((ref.piece_id, ref.take_id))
            if tk is None or tk.task != anchor_task:
                continue
            if ref.take_id == exclude_take:
                continue
            est_start = q_time - ref.anchor_time
            if est_start < 0:
                continue
            bin_idx = int(est_start / OFFSET_BIN_SEC)
            key = (ref.piece_id, ref.take_id, bin_idx)
            votes[key] += 1
            sums[key] += est_start

    seen_pieces: dict[str, dict] = {}
    for (pid, tid, _), cnt in votes.items():
        tk = meta.get((pid, tid))
        if tk is None or tk.n_fingerprints == 0:
            continue
        threshold = max(MIN_VOTES_FLOOR, int(tk.n_fingerprints * MIN_RATIO))
        if cnt < threshold:
            continue
        ratio = cnt / tk.n_fingerprints
        rec = {
            "piece_id": pid,
            "take_id": tid,
            "score": round(ratio, 4),
            "matched": cnt,
            "anchor_fps": tk.n_fingerprints,
            "anchor_duration_sec": round(tk.duration_sec, 3),
            "anchor_task": tk.task,
            "query_offset_sec": round(sums[(pid, tid, _)] / cnt, 3),
        }
        prev = seen_pieces.get(pid)
        if prev is None or rec["score"] > prev["score"]:
            seen_pieces[pid] = rec

    return sorted(seen_pieces.values(), key=lambda r: -r["score"])[:TOP_K]


def main():
    print(f"Building anchor DB from {ANCHOR_ROOT}...")
    db, meta = build_normalized_db()
    if not meta:
        return

    # Per-task anchor coverage summary
    by_task = defaultdict(int)
    by_task_pieces = defaultdict(set)
    for (pid, tid), tk in meta.items():
        by_task[tk.task] += 1
        by_task_pieces[tk.task].add(pid)
    print("Anchor coverage:")
    for t in ("Technique1", "Technique2", "Set1", "Free"):
        n_takes = by_task[t]
        n_pieces = len(by_task_pieces[t])
        print(f"  {t:12s}: {n_pieces:3d} pieces, {n_takes:3d} takes")

    if not PIECE_ROOT.is_dir():
        print(f"  ERROR: {PIECE_ROOT} not found")
        return

    pieces: dict = {}
    n_processed = 0
    n_matched = 0
    skipped_no_pool = 0
    by_task_match = {0: [0,0], 1: [0,0], 2: [0,0], 9: [0,0]}  # [matched, total]

    files = sorted(f for f in PIECE_ROOT.iterdir()
                   if f.is_file() and f.name.endswith(".mid"))
    for f in files:
        m = PIECE_FILE_RE.match(f.name)
        if not m:
            continue
        trial_stem = m.group(1)
        task = int(m.group(2))
        piece_idx = int(m.group(3))
        anchor_task = TASK_TO_ANCHOR.get(task)
        if anchor_task is None:
            skipped_no_pool += 1
            continue
        # Exclude this piece's own take from the match pool (otherwise a
        # confirmed piece would always 100%-match itself).
        exclude_take = f"{trial_stem}_p{piece_idx}"

        try:
            cands = match_piece(f, db, meta, anchor_task, exclude_take)
        except Exception as e:
            print(f"  [WARN] match failed on {f.name}: {e}")
            continue

        key = f"{trial_stem}::{piece_idx}"
        pieces[key] = {
            "trial_stem": trial_stem,
            "piece_idx": piece_idx,
            "task": task,
            "anchor_task": anchor_task,
            "candidates": cands,
        }
        n_processed += 1
        by_task_match[task][1] += 1
        if cands:
            n_matched += 1
            by_task_match[task][0] += 1

        if n_processed % 50 == 0:
            print(f"  processed {n_processed}/{len(files)} pieces, "
                  f"{n_matched} with ≥1 candidate")

    print(f"\nDone: {n_processed} pieces processed "
          f"({n_matched} with candidates, {n_processed - n_matched} unmatched).")
    print("By task:")
    name = {0: "T0 Tech1", 1: "T1 Tech2", 2: "T2 Set1", 9: "T9 Free"}
    for t in (0, 1, 2, 9):
        m_, tot = by_task_match[t]
        print(f"  {name[t]:10s}: {m_:3d}/{tot:3d} matched")

    out = {
        "_comment": "Per-piece fingerprint matches against 240828 self-anchors only "
                    "(Data/Anchors/2024). 220819 anchors NOT used — single-session "
                    "consistency is more reliable for cluster signal.",
        "anchor_root": str(ANCHOR_ROOT).replace("\\", "/"),
        "min_ratio": MIN_RATIO,
        "min_votes_floor": MIN_VOTES_FLOOR,
        "top_k": TOP_K,
        "n_anchor_takes": len(meta),
        "n_pieces": n_processed,
        "n_matched": n_matched,
        "pieces": pieces,
    }
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, "w", encoding="utf-8") as fh:
        json.dump(out, fh, ensure_ascii=False, indent=2)
    print(f"\nSaved: {OUT_PATH}")


if __name__ == "__main__":
    main()
