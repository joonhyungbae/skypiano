"""
Piece windows the reviewer set by hand.

The review page saves them to Data/Metadata/{ds}/_window_overrides.json as
{piece_stem: {start, end, at}} (trial seconds), together with a "yes" verdict
on that window. The review page snaps the edges to the piece's first and last
notes, so a set window is the piece's core: Stage 04 puts it in place of the
computed core of the piece it overlaps most (piece numbers may shift when a
trial is cut again) and pads it like any other piece.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

META = Path("Data/Metadata")


def load(ds: str) -> dict[str, list[tuple[float, float]]]:
    """trial stem -> set windows."""
    f = META / ds / "_window_overrides.json"
    if not f.exists() or os.environ.get("SKIP_WINDOW_OVERRIDES"):      # to compare the computed windows
        return {}
    out: dict[str, list[tuple[float, float]]] = {}
    for stem, o in json.loads(f.read_text(encoding="utf-8")).items():
        out.setdefault(stem.rsplit("_p", 1)[0], []).append((float(o["start"]), float(o["end"])))
    return out


def apply(windows: list[tuple[float, float]], set_windows: list[tuple[float, float]]) -> tuple[list[tuple[float, float]], set[int]]:
    """Cores with the set ones in place of those they overlap most; and which were replaced."""
    out, done = list(windows), set()
    for a, b in set_windows:
        best, k = 0.0, None
        for i, (x, y) in enumerate(out):
            ov = min(b, y) - max(a, x)
            if ov > best and i not in done:
                best, k = ov, i
        if k is not None:
            out[k] = (round(a, 3), round(b, 3))
            done.add(k)
    return out, done
