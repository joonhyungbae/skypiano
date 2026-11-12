"""
Piece boundaries confirmed by content: does a cut piece hold exactly its
piece, as the other performers of the same piece played it?

Every performer of a named piece played the same score. The instances whose
chord count is within REF_COUNT_TOL of the piece's median give a reference
(their medoid by chord-sequence edit distance). An instance's boundaries
match by content when, against that reference,

  * the whole piece aligns (score >= MATCH_MIN, chords matching at Jaccard
    >= 0.5),
  * the cut begins with the reference's first EDGE_CHORDS chords and ends
    with its last EDGE_CHORDS chords, and
  * no chord of the cut lies outside the aligned span.

This is a check of the cut against the music, not an estimate; a piece that
fails it is left for review by eye. Pieces whose name has fewer than
MIN_INSTANCES performances (free pieces) cannot be checked.

Chords are notes starting within CHORD_S of a chord's first note.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pretty_midi

CHORD_S = 0.05
REF_COUNT_TOL = 0.03
MATCH_MIN = 0.9
EDGE_CHORDS = 3
MIN_INSTANCES = 3


def chords(midi: Path) -> list[frozenset]:
    ns = sorted((n.start, n.pitch) for i in pretty_midi.PrettyMIDI(str(midi)).instruments for n in i.notes)
    ev: list[list] = []
    for t, p in ns:
        if ev and t - ev[-1][0] <= CHORD_S:
            ev[-1][1].add(p)
        else:
            ev.append([t, {p}])
    return [frozenset(p) for _, p in ev]


def _same(x: frozenset, y: frozenset) -> bool:
    return len(x & y) * 2 >= len(x | y)


def _edit_distance(a: list, b: list) -> int:
    d = list(range(len(b) + 1))
    for i in range(1, len(a) + 1):
        prev, d[0] = d[:], i
        for j in range(1, len(b) + 1):
            d[j] = min(prev[j] + 1, d[j - 1] + 1, prev[j - 1] + (0 if _same(a[i - 1], b[j - 1]) else 1))
    return d[-1]


def _align(ref: list, tgt: list) -> tuple[float, int, int]:
    """Whole `ref` against a free stretch of `tgt`: (score, first, last aligned index in tgt)."""
    n, m = len(ref), len(tgt)
    H = np.zeros((n + 1, m + 1))
    H[1:, 0] = -np.arange(1, n + 1)
    S = np.zeros((n + 1, m + 1), dtype=int)
    S[0, :] = np.arange(m + 1)
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            c = (H[i - 1, j - 1] + (2 if _same(ref[i - 1], tgt[j - 1]) else -1), H[i - 1, j] - 1, H[i, j - 1] - 1)
            k = int(np.argmax(c))
            H[i, j] = c[k]
            S[i, j] = S[i - 1, j - 1] if k == 0 else (S[i - 1, j] if k == 1 else S[i, j - 1])
    j = int(np.argmax(H[n, 1:])) + 1
    return float(H[n, j] / (2 * n)), int(S[n, j]), j - 1


def content_matches(pieces: list[tuple[str, str, Path]]) -> dict[str, dict]:
    """pieces: (piece_stem, name, midi path) -> {piece_stem: {match, score, n_ref}} for checkable names."""
    groups: dict[str, list[tuple[str, list]]] = {}
    for stem, name, midi in pieces:
        if name and midi.exists():
            groups.setdefault(name, []).append((stem, chords(midi)))
    out = {}
    for items in groups.values():
        if len(items) < MIN_INSTANCES:
            continue
        med = float(np.median([len(c) for _, c in items]))
        cand = [c for _, c in items if abs(len(c) - med) <= max(1, REF_COUNT_TOL * med)]
        if len(cand) < 2 or med < 2 * EDGE_CHORDS:
            continue
        ref = min(cand, key=lambda x: sum(_edit_distance(x, y) for y in cand[:6]))
        for stem, c in items:
            if len(c) < EDGE_CHORDS:
                out[stem] = {"match": False, "score": 0.0, "n_ref": len(ref)}
                continue
            score, i0, i1 = _align(ref, c)
            edges = all(_same(ref[k], c[k]) for k in range(EDGE_CHORDS)) and \
                all(_same(ref[-1 - k], c[-1 - k]) for k in range(EDGE_CHORDS))
            ok = score >= MATCH_MIN and edges and i0 == 0 and i1 == len(c) - 1
            out[stem] = {"match": bool(ok), "score": round(score, 3), "n_ref": len(ref)}
    return out
