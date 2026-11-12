"""
Piece cuts checked against the score.

The technique drills (SET 2 of the technique book, 240828 task 0) are written
out from their scores (Stage 04 s14: set2_score, t0_score). A cut of one of
them holds exactly that drill when

  * the cut, from its first chord to its last, aligns with the whole score,
    first to first and last to last (_common/score_follow.whole_cost on onset
    frames), at mean cost <= MAX_COST,
  * its first and last chords are the score's first and last (cosine of the
    onset frames >= EDGE_MATCH), and
  * nothing else is in the cut (anything else would break the alignment).

Pieces without a written score (the Fink exercises) are checked the same way
against performances of them confirmed elsewhere (by review, by content or by
the score), in any dataset: performance_match passes when the cut matches one
of them whole to whole at mean cost <= MAX_COST with the same first and last
chords.

A cut that misses these only by playing slips -- whole to whole at mean cost
<= NEAR_COST, its first and last chords at cosine >= NEAR_EDGE with the
reference's -- is a near match (near_match), kept apart from the exact ones.

Octave repeats are not checked: the same chord repeated, a cut that misses
part of the run still matches. This is a check of the notes against the score, not an estimate; a
cut that fails it is asked about.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pretty_midi

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _common.score_follow import onset_frames, whole_cost  # noqa: E402

MAX_COST = 0.1
EDGE_MATCH = 0.9
# a near match: the right stretch played with slips (a note missed or added) -- whole to
# whole at mean cost <= NEAR_COST, first and last chords sharing most of their notes
NEAR_COST = 0.15
NEAR_EDGE = 0.7
_STEPS = Path(__file__).resolve().parents[1] / "04_piece_split" / "steps"
_SCORES: dict | None = None


def _scores() -> dict:
    global _SCORES
    if _SCORES is None:
        spec = importlib.util.spec_from_file_location("s14_240828_program_pieces",
                                                      _STEPS / "s14_240828_program_pieces.py")
        s14 = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = s14
        spec.loader.exec_module(s14)
        _SCORES = {**{k: onset_frames(v)[0] for k, v in s14.set2_score().items()},
                   **{k: onset_frames(v)[0] for k, v in s14.t0_score().items()},
                   "Octave Repeat x5": onset_frames(s14.octave_repeat(40))[0]}
    return _SCORES


def score_for(name: str | None) -> np.ndarray | None:
    """The score's frames for a piece name in the words of the score, or None. An octave
    repeat is the same chord over and over: a cut missing some of the run at either end
    still matches, so octave repeats are not checked (None)."""
    if not name or name.startswith("Octave Repeat"):
        return None
    sc = _scores()
    if name in sc:
        return sc[name]
    base = name.split(" (")[0]
    if base == "Octave Repeat":
        return sc["Octave Repeat x5"] if "x5" in name else sc["Octave Repeat"]
    return sc.get(base) if base in ("Scale", "Arpeggio", "Octave Scale", "Broken Octave Arpeggio") else None


def score_match(midi: Path, name: str | None) -> dict | None:
    """{match, cost} for a cut whose name has a score, else None."""
    R = score_for(name)
    if R is None or not midi.exists():
        return None
    notes = [(n.start, n.end, n.pitch) for i in pretty_midi.PrettyMIDI(str(midi)).instruments for n in i.notes]
    F = onset_frames(notes)[0]
    if len(F) < 2:
        return {"match": False, "cost": None}
    c = whole_cost(R, F)
    edges = float(F[0] @ R[0]) >= EDGE_MATCH and float(F[-1] @ R[-1]) >= EDGE_MATCH
    if not np.isfinite(c):
        return {"match": False, "cost": None}
    return {"match": bool(edges and c <= MAX_COST), "cost": round(c, 3)}


def cut_frames(midi: Path) -> np.ndarray:
    notes = [(n.start, n.end, n.pitch) for i in pretty_midi.PrettyMIDI(str(midi)).instruments for n in i.notes]
    return onset_frames(notes)[0]


def whole_match(F: np.ndarray, R: np.ndarray, edge: float = EDGE_MATCH) -> float | None:
    """Mean cost of the whole cut against the whole reference when their first and last
    chords agree (cosine >= edge), else None."""
    if len(F) < 2 or len(R) < 2 or float(F[0] @ R[0]) < edge or float(F[-1] @ R[-1]) < edge:
        return None
    c = whole_cost(R, F)
    return c if np.isfinite(c) else None


def performance_match(midi: Path, name: str | None, confirmed: list[np.ndarray]) -> dict | None:
    """{match, cost} against confirmed performances of the same piece, else None."""
    if not name or name.startswith("Octave Repeat") or not confirmed or not midi.exists():
        return None
    F = cut_frames(midi)
    costs = [c for R in confirmed if (c := whole_match(F, R)) is not None]
    best = min(costs) if costs else None
    return {"match": best is not None and best <= MAX_COST, "cost": None if best is None else round(best, 3)}


def near_match(midi: Path, name: str | None, confirmed: list[np.ndarray]) -> dict | None:
    """{match, cost}: the cut against the score and confirmed performances, slips allowed."""
    if not name or name.startswith("Octave Repeat") or not midi.exists():
        return None
    refs = list(confirmed)
    sc = score_for(name)
    if sc is not None:
        refs.append(sc)
    if not refs:
        return None
    F = cut_frames(midi)
    costs = [c for R in refs if (c := whole_match(F, R, NEAR_EDGE)) is not None]
    best = min(costs) if costs else None
    return {"match": best is not None and best <= NEAR_COST, "cost": None if best is None else round(best, 3)}
