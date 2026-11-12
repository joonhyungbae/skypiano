"""
MIDI symbolic fingerprint matcher.

Detect WHERE each anchor piece occurs within a long query MIDI by hashing
pitch-class + inter-onset-interval (IOI) n-grams. Matches are grouped by
consistent (anchor_id, time_offset) clusters.

Why this approach (vs Shazam audio):
  - MIDI keyboard capture is a clean symbolic signal — no spectral noise.
  - Pitch-interval + IOI-ratio fingerprints are tempo-/transposition-invariant,
    so the same piece played by a different performer still matches.
  - O(notes) extraction; matching is hash-table lookup.

Reference: Wang 2003 (Shazam constellation hashing) adapted for symbolic MIDI;
similar in spirit to Nakamura HMM but lighter.

Usage:
    from src.midi_fingerprint import build_db, match
    db = build_db("Data/Anchors")
    hits = match("Data/Raw/240828/.../session.mid", db)
    # hits = [(piece_id, take_id, query_start_sec, score), ...]
"""
from __future__ import annotations

import json
import math
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import pretty_midi


_YEAR_RE = re.compile(r"^\d{4}$")


# ────────────────────────────────────────────────────────────────
# Note extraction
# ────────────────────────────────────────────────────────────────

@dataclass
class Event:
    """A musical event = chord (set of simultaneous pitches) at onset time."""
    onset: float
    pitches: tuple[int, ...]   # sorted ascending

    @property
    def top(self) -> int:
        return self.pitches[-1]

    @property
    def bass(self) -> int:
        return self.pitches[0]


CHORD_WINDOW = 0.04   # notes within 40 ms of each other → one chord event


def events_from_notes(raw: list[tuple[float, int]]) -> list[Event]:
    """Group near-simultaneous (onset, pitch) pairs into chord events."""
    if not raw:
        return []
    raw = sorted(raw)
    events: list[Event] = []
    cur_onset = raw[0][0]
    cur_pitches: list[int] = [raw[0][1]]
    for t, p in raw[1:]:
        if t - cur_onset <= CHORD_WINDOW:
            cur_pitches.append(p)
        else:
            events.append(Event(cur_onset, tuple(sorted(set(cur_pitches)))))
            cur_onset = t
            cur_pitches = [p]
    events.append(Event(cur_onset, tuple(sorted(set(cur_pitches)))))
    return events


def extract_events(midi_path: Path) -> list[Event]:
    """Group near-simultaneous notes into chord events sorted by onset."""
    pm = pretty_midi.PrettyMIDI(str(midi_path))
    raw: list[tuple[float, int]] = []
    for inst in pm.instruments:
        if inst.is_drum:
            continue
        for n in inst.notes:
            raw.append((n.start, n.pitch))
    return events_from_notes(raw)


# ────────────────────────────────────────────────────────────────
# Fingerprint hashing
# ────────────────────────────────────────────────────────────────

# Quantize IOI ratio to discrete bins so small timing variations still hash to
# the same bucket. Bin = round(log2(ratio) * BIN_RES).
IOI_BIN_RES = 4   # bins per octave-of-ratio (so adjacent bins differ by ×2^(1/4))
NGRAM = 3         # events per fingerprint window (3 events ≈ 2 IOIs + 3 pitches);
                  # short T1 pieces have only ~7 events, so 5-grams under-cover.
MIN_IOI = 0.02    # ignore intervals shorter than 20 ms (dedup chord)


def _ioi_bin(ratio: float) -> int:
    """Map IOI ratio to integer bin (tempo-invariant)."""
    if ratio <= 0:
        return 0
    return int(round(math.log2(ratio) * IOI_BIN_RES))


def fingerprint(events: list[Event], ngram: int = NGRAM) -> list[tuple[int, float]]:
    """Return list of (hash, event_time) for sliding n-gram windows over chord events.

    Each fingerprint encodes (transposition + tempo invariant):
      - top-line pitch intervals between consecutive events
      - bass-line pitch intervals (captures harmonic motion)
      - IOI ratio pattern between consecutive events
    """
    if len(events) < ngram:
        return []

    out: list[tuple[int, float]] = []
    for i in range(len(events) - ngram + 1):
        window = events[i:i + ngram]
        iois = [window[k + 1].onset - window[k].onset for k in range(ngram - 1)]
        if min(iois) < MIN_IOI:
            continue

        top_iv = tuple(window[k + 1].top - window[k].top for k in range(ngram - 1))
        bass_iv = tuple(window[k + 1].bass - window[k].bass for k in range(ngram - 1))

        base = iois[0]
        if base < MIN_IOI:
            continue
        ratios = tuple(_ioi_bin(iois[k] / base) for k in range(ngram - 1))

        h = hash((top_iv, bass_iv, ratios))
        out.append((h, window[0].onset))

    return out


# ────────────────────────────────────────────────────────────────
# Anchor DB
# ────────────────────────────────────────────────────────────────

@dataclass
class AnchorRef:
    piece_id: str
    take_id: str
    anchor_time: float  # onset of fingerprint within the anchor MIDI


@dataclass
class AnchorMeta:
    n_fingerprints: int     # how many windows this take produced
    duration_sec: float     # total length of the anchor


def build_db(anchor_root: Path | str = "Data/Anchors",
             task_filter: set[str] | None = None,
             verbose: bool = True) -> tuple[
        dict[int, list[AnchorRef]], dict[tuple[str, str], AnchorMeta]]:
    """Build hash → [AnchorRef] DB plus per-take metadata from anchor library.

    Layout: anchor_root may contain task_dirs directly (Data/Anchors/Technique1/…)
    OR year-prefixed task_dirs (Data/Anchors/2022/Technique1/…). Both forms are
    walked transparently — useful since reviewer-confirmed 240828 anchors live
    under 2024/ while older or future bootstraps may sit elsewhere.

    task_filter: if given, only include takes whose task_dir.name is in this set
    (e.g. {"Set1"}). None = include all tasks.
    """
    anchor_root = Path(anchor_root)
    db: dict[int, list[AnchorRef]] = defaultdict(list)
    meta: dict[tuple[str, str], AnchorMeta] = {}
    n_takes = 0

    def ingest_task_dir(task_dir: Path) -> None:
        nonlocal n_takes
        if task_filter is not None and task_dir.name not in task_filter:
            return
        for piece_dir in task_dir.iterdir():
            if not piece_dir.is_dir():
                continue
            piece_id = piece_dir.name
            for mid in piece_dir.glob("*.mid"):
                take_id = mid.stem
                events = extract_events(mid)
                fps = fingerprint(events)
                for h, t in fps:
                    db[h].append(AnchorRef(piece_id, take_id, t))
                meta[(piece_id, take_id)] = AnchorMeta(
                    n_fingerprints=len(fps),
                    duration_sec=events[-1].onset - events[0].onset if events else 0.0,
                )
                n_takes += 1

    for entry in anchor_root.iterdir():
        if not entry.is_dir():
            continue
        if _YEAR_RE.match(entry.name):
            for task_dir in entry.iterdir():
                if task_dir.is_dir():
                    ingest_task_dir(task_dir)
        else:
            ingest_task_dir(entry)

    if verbose:
        filter_str = f" [filter={sorted(task_filter)}]" if task_filter else ""
        print(f"[anchor-db]{filter_str} {n_takes} takes → {len(db)} unique hashes "
              f"(avg {sum(len(v) for v in db.values()) / max(len(db),1):.1f} refs/hash)")
    return db, meta


# ────────────────────────────────────────────────────────────────
# Matching
# ────────────────────────────────────────────────────────────────

@dataclass
class Hit:
    piece_id: str
    take_id: str
    query_start_sec: float   # estimated onset of this piece in the query
    anchor_duration_sec: float
    n_votes: int             # number of fingerprints voting for this offset
    hit_ratio: float         # n_votes / anchor_n_fingerprints (0..1; ~1 = full match)


def match(query_midi: Path | str,
          db: dict[int, list[AnchorRef]],
          meta: dict[tuple[str, str], AnchorMeta],
          *,
          min_ratio: float = 0.30,    # fraction of anchor fingerprints that must match
          min_votes_floor: int = 2,    # absolute minimum (very short anchors)
          offset_bin_sec: float = 0.5,
          dedup_window_sec: float = 5.0,
          ) -> list[Hit]:
    """Match a query MIDI against the anchor DB.

    Algorithm:
      1. Fingerprint the query.
      2. For each (q_hash, q_time), look up anchor refs; each ref votes for
         offset = q_time - anchor_time (= where the anchor would start in query).
      3. Bin votes by (piece_id, take_id, q_start // bin), keep clusters where
         votes ≥ max(min_votes_floor, anchor_fp_count × min_ratio).
      4. For each piece, keep best take.
      5. De-duplicate temporally: within a sliding window only keep top-vote hit
         (avoids cascade of partial matches near a true onset).
    """
    query_events = extract_events(Path(query_midi))
    query_fps = fingerprint(query_events)

    # vote: (piece_id, take_id, bin_index) -> (count, sum_q_start)
    votes: dict[tuple[str, str, int], tuple[int, float]] = defaultdict(lambda: (0, 0.0))

    for q_hash, q_time in query_fps:
        if q_hash not in db:
            continue
        for ref in db[q_hash]:
            est_start = q_time - ref.anchor_time
            if est_start < 0:
                continue
            bin_idx = int(est_start / offset_bin_sec)
            cnt, sm = votes[(ref.piece_id, ref.take_id, bin_idx)]
            votes[(ref.piece_id, ref.take_id, bin_idx)] = (cnt + 1, sm + est_start)

    hits: list[Hit] = []
    for (pid, tid, _), (cnt, sm) in votes.items():
        m = meta.get((pid, tid))
        if m is None or m.n_fingerprints == 0:
            continue
        threshold = max(min_votes_floor, int(m.n_fingerprints * min_ratio))
        if cnt < threshold:
            continue
        hits.append(Hit(
            piece_id=pid,
            take_id=tid,
            query_start_sec=sm / cnt,
            anchor_duration_sec=m.duration_sec,
            n_votes=cnt,
            hit_ratio=cnt / m.n_fingerprints,
        ))

    # Per piece, keep best take (highest hit_ratio) within the same vicinity.
    # Temporal de-dup: scan sorted by ratio desc, keep hit if no higher-ratio
    # hit already accepted within ±dedup_window_sec.
    accepted: list[Hit] = []
    for h in sorted(hits, key=lambda x: -x.hit_ratio):
        if any(abs(h.query_start_sec - a.query_start_sec) <= dedup_window_sec
               and h.piece_id == a.piece_id for a in accepted):
            continue
        accepted.append(h)

    return sorted(accepted, key=lambda h: h.query_start_sec)


# ────────────────────────────────────────────────────────────────
# CLI
# ────────────────────────────────────────────────────────────────

def main():
    import sys
    if len(sys.argv) < 2:
        print("usage: python midi_fingerprint.py <query.mid> [--ratio R]")
        sys.exit(0)
    query = sys.argv[1]
    min_ratio = 0.30
    for i, a in enumerate(sys.argv):
        if a == "--ratio" and i + 1 < len(sys.argv):
            min_ratio = float(sys.argv[i + 1])

    db, meta = build_db()
    hits = match(query, db, meta, min_ratio=min_ratio)
    print(f"\n=== Matches in {Path(query).name} (min_ratio={min_ratio}) ===")
    print(f"{'t (sec)':>9}  {'dur':>6}  {'votes':>5}  {'ratio':>5}  piece                          take")
    for h in hits:
        print(f"  {h.query_start_sec:7.2f}  {h.anchor_duration_sec:6.2f}  "
              f"{h.n_votes:5d}  {h.hit_ratio:5.2f}  {h.piece_id!r:30s} {h.take_id}")


if __name__ == "__main__":
    main()
