"""
Split a long MIDI session into piece/take-level segments by silence gaps.

Why segment first:
  Full-session fingerprint matching suffers from partial-match noise — short T1
  pieces produce too few n-grams to disambiguate. After splitting the session
  into phrases, each segment can be fingerprinted as a whole and compared
  against anchor DB piece-by-piece (much higher SNR).

Segmentation rule:
  gap = next_note.start - running_max(end_of_prior_notes)
  If gap >= GAP_THRESHOLD, close the current segment and start a new one.
  Segments shorter than MIN_DURATION (or with fewer than MIN_NOTES notes) are
  dropped as noise (single stray notes, warm-up taps).
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pretty_midi


GAP_THRESHOLD = 3.0   # sec; picked from the 240828 inter-onset gap distribution
MIN_DURATION = 1.5    # sec; drop micro-segments
MIN_NOTES = 8


@dataclass
class Segment:
    start: float          # sec (session time)
    end: float            # sec
    n_notes: int
    notes: list[tuple[float, int]]  # (onset, pitch) sorted

    @property
    def duration(self) -> float:
        return self.end - self.start


def segment_midi(midi_path: Path | str,
                 *,
                 gap_threshold: float = GAP_THRESHOLD,
                 min_duration: float = MIN_DURATION,
                 min_notes: int = MIN_NOTES,
                 ) -> list[Segment]:
    pm = pretty_midi.PrettyMIDI(str(midi_path))
    raw: list[tuple[float, float, int]] = []  # (start, end, pitch)
    for inst in pm.instruments:
        if inst.is_drum:
            continue
        for n in inst.notes:
            raw.append((n.start, n.end, n.pitch))
    raw.sort()
    if not raw:
        return []

    segments: list[Segment] = []
    cur: list[tuple[float, float, int]] = [raw[0]]
    running_end = raw[0][1]

    def _flush():
        if not cur:
            return
        start = cur[0][0]
        end = max(n[1] for n in cur)
        notes = [(s, p) for s, _, p in cur]
        if end - start >= min_duration and len(notes) >= min_notes:
            segments.append(Segment(
                start=start,
                end=end,
                n_notes=len(notes),
                notes=notes,
            ))

    for s, e, p in raw[1:]:
        gap = s - running_end
        if gap >= gap_threshold:
            _flush()
            cur = [(s, e, p)]
            running_end = e
        else:
            cur.append((s, e, p))
            running_end = max(running_end, e)
    _flush()
    return segments


def write_segment_midi(seg: Segment, src_midi: Path | str, dst: Path) -> None:
    """Extract segment [seg.start, seg.end] from src MIDI and save to dst."""
    pm = pretty_midi.PrettyMIDI(str(src_midi))
    new_pm = pretty_midi.PrettyMIDI(initial_tempo=120.0)
    for inst in pm.instruments:
        if inst.is_drum:
            continue
        new_inst = pretty_midi.Instrument(program=inst.program, is_drum=False, name=inst.name)
        for n in inst.notes:
            if n.start >= seg.start and n.start < seg.end:
                new_inst.notes.append(pretty_midi.Note(
                    velocity=n.velocity,
                    pitch=n.pitch,
                    start=n.start - seg.start,
                    end=min(n.end, seg.end) - seg.start,
                ))
        if new_inst.notes:
            new_pm.instruments.append(new_inst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    new_pm.write(str(dst))


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("usage: python midi_segment.py <midi> [--gap 3.0]")
        sys.exit(0)
    mid = sys.argv[1]
    gap = GAP_THRESHOLD
    for i, a in enumerate(sys.argv):
        if a == "--gap" and i + 1 < len(sys.argv):
            gap = float(sys.argv[i + 1])

    segs = segment_midi(mid, gap_threshold=gap)
    print(f"\n=== {Path(mid).name} (gap>={gap}s): {len(segs)} segments ===")
    print(f"{'idx':>3}  {'start':>7}  {'end':>7}  {'dur':>6}  {'notes':>6}")
    for i, s in enumerate(segs):
        print(f"{i:>3}  {s.start:>7.2f}  {s.end:>7.2f}  {s.duration:>6.2f}  {s.n_notes:>6d}")
