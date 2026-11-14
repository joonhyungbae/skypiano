"""
Piece windows the reviewer set by hand.

The review page saves them to Data/Metadata/{ds}/_window_overrides.json as
{piece_stem: {start, end, at}} (trial seconds), together with a "yes" verdict
on that window. The review page snaps the edges to the piece's first and last
notes, so a set window is the piece's core: Stage 04 puts it in place of the
computed core of the piece it overlaps most (piece numbers may shift when a
trial is cut again) and pads it like any other piece. A set window that
overlaps no computed core (the piece was played in full only later in the
trial, after false starts) replaces the core of the piece its stem names.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

META = Path("Data/Metadata")


def load(ds: str) -> dict[str, list[tuple[float, float, int]]]:
    """trial stem -> set windows (start, end, piece number of the stem)."""
    f = META / ds / "_window_overrides.json"
    if not f.exists() or os.environ.get("SKIP_WINDOW_OVERRIDES"):      # to compare the computed windows
        return {}
    out: dict[str, list[tuple[float, float, int]]] = {}
    for stem, o in json.loads(f.read_text(encoding="utf-8")).items():
        trial, k = stem.rsplit("_p", 1)
        out.setdefault(trial, []).append((float(o["start"]), float(o["end"]), int(k)))
    return out


def apply(windows: list[tuple[float, float]], set_windows: list[tuple]) -> tuple[list[tuple[float, float]], set[int]]:
    """Cores with the set ones in place of those they overlap most (else of the
    piece their stem names); and which were replaced."""
    out, done = list(windows), set()
    for a, b, *num in set_windows:
        best, k = 0.0, None
        for i, (x, y) in enumerate(out):
            ov = min(b, y) - max(a, x)
            if ov > best and i not in done:
                best, k = ov, i
        if k is None and num and 1 <= num[0] <= len(out) and num[0] - 1 not in done:
            k = num[0] - 1
        if k is not None:
            out[k] = (round(a, 3), round(b, 3))
            done.add(k)
    return out, done
