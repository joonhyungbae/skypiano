#!/usr/bin/env python3
"""
Build body_segments.csv (and optional body_segment_centers.csv) from a
flat-marker body CSV.

Input CSV header is flat marker columns (no Frame/Time prefix):
  HEADT_x,HEADT_y,HEADT_z,HEADA_x,...,LSHANK2_x,LSHANK2_y,LSHANK2_z

The 240828 piano session uses the "piano" marker naming, so we reuse
remap_piano_markers + compute_segments_piano from s02_segment_piano_raw.py.

Output CSV header:
  head_a_x,head_a_y,head_a_z,head_b_x,...,left_foot_b_z

13 segments x {a,b} x {x,y,z} = 78 columns. Each segment is represented by
its two endpoints (a, b); column keys follow the '<marker>_x|y|z' convention.
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from typing import Dict, List

import numpy as np

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)

from s03_segment_qtm_raw import SEGMENTS
from s02_segment_piano_raw import compute_segments_piano, remap_piano_markers


# 240828 / Explorer body CSV uses RTOEF/LTOEF for the toe-front (meta-2-equivalent)
# marker, while s02_segment_piano_raw.PIANO_TO_CANONICAL only knows RTOEM/LTOEM.
# Without this alias, right_foot_b = mean(RHEEL, NaN) collapses to RHEEL, drawing
# the foot stick backwards. Aliasing pre-remap fixes the foot direction.
EXPLORER_TO_PIANO_ALIASES: Dict[str, str] = {
    "RTOEF": "RTOEM",
    "LTOEF": "LTOEM",
}


def _apply_explorer_aliases(raw: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    out = dict(raw)
    for src, dst in EXPLORER_TO_PIANO_ALIASES.items():
        if src in raw and dst not in raw:
            out[dst] = raw[src].copy()
    return out


def parse_explorer_body_csv(path: str):
    with open(path, "r", newline="") as f:
        reader = csv.reader(f)
        header = next(reader)
        marker_cols: Dict[str, Dict[str, int]] = {}
        for i, col in enumerate(header):
            col = col.strip()
            if "_" not in col:
                continue
            name, axis = col.rsplit("_", 1)
            axis = axis.lower()
            if axis not in ("x", "y", "z"):
                continue
            marker_cols.setdefault(name, {})[axis] = i

        markers = [m for m, ax in marker_cols.items() if {"x", "y", "z"} <= ax.keys()]

        for row_idx, row in enumerate(reader):
            raw: Dict[str, np.ndarray] = {}
            for m in markers:
                ix = marker_cols[m]
                try:
                    x = float(row[ix["x"]])
                    y = float(row[ix["y"]])
                    z = float(row[ix["z"]])
                except (ValueError, IndexError):
                    x = y = z = float("nan")
                raw[m] = np.array([x, y, z], dtype=float)
            yield row_idx, raw


def write_body_segments_csv(in_path: str, out_path: str, with_centers: bool = True) -> None:
    seg_fields: List[str] = []
    for seg in SEGMENTS:
        for end in ("a", "b"):
            for axis in ("x", "y", "z"):
                seg_fields.append(f"{seg}_{end}_{axis}")
    # Head anterior reference (raw forehead marker) for orienting the head's
    # forward axis in viewers — Visual3D head segment has its anterior axis
    # defined by HEAD_FORHEAD, which is not one of the segment's endpoints.
    for axis in ("x", "y", "z"):
        seg_fields.append(f"head_front_{axis}")

    centers_path = None
    if with_centers:
        base, ext = os.path.splitext(out_path)
        centers_path = base.replace("body_segments", "body_segment_centers") + ext
        if centers_path == out_path:
            centers_path = base + "_centers" + ext

    center_fields: List[str] = []
    for seg in SEGMENTS:
        for axis in ("x", "y", "z"):
            center_fields.append(f"{seg}_c_{axis}")

    out_dir = os.path.dirname(out_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    fseg = open(out_path, "w", newline="")
    fc = open(centers_path, "w", newline="") if with_centers else None
    try:
        wseg = csv.writer(fseg)
        wseg.writerow(seg_fields)
        wc = csv.writer(fc) if fc else None
        if wc:
            wc.writerow(center_fields)

        for _frame_idx, raw in parse_explorer_body_csv(in_path):
            aliased = _apply_explorer_aliases(raw)
            m = remap_piano_markers(aliased)
            segs = compute_segments_piano(m)

            seg_row: List[str] = []
            center_row: List[str] = []
            for seg in SEGMENTS:
                a = segs[f"{seg}_a"]
                b = segs[f"{seg}_b"]
                for axis_idx in range(3):
                    seg_row.append(_fmt(a[axis_idx]))
                for axis_idx in range(3):
                    seg_row.append(_fmt(b[axis_idx]))
                if wc:
                    c = (a + b) / 2.0
                    for axis_idx in range(3):
                        center_row.append(_fmt(c[axis_idx]))
            head_front = m.get("HEAD_FORHEAD")
            if head_front is None:
                head_front = np.array([np.nan, np.nan, np.nan])
            for axis_idx in range(3):
                seg_row.append(_fmt(head_front[axis_idx]))
            wseg.writerow(seg_row)
            if wc:
                wc.writerow(center_row)
    finally:
        fseg.close()
        if fc:
            fc.close()


def _fmt(v: float) -> str:
    if not np.isfinite(v):
        return ""
    return f"{v:.3f}"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="in_path", required=True, help="Input Explorer body CSV")
    ap.add_argument("--out", dest="out_path", default=None, help="Output body_segments.csv (default: <in_dir>/body_segments.csv)")
    ap.add_argument("--no-centers", action="store_true", help="Skip writing body_segment_centers.csv")
    args = ap.parse_args()

    in_path = args.in_path
    out_path = args.out_path or os.path.join(os.path.dirname(in_path), "body_segments.csv")
    write_body_segments_csv(in_path, out_path, with_centers=not args.no_centers)
    print("WROTE", out_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
