#!/usr/bin/env python3
"""
Build '*_body_segments.tsv' from Qualisys piano-style TSV (marker names differ from S5).

Piano-specific rules (per user):
- No fingertip / meta2: hand segment is wrist -> hand marker only (no RFINGERTIP/LFINGERTIP).
- Toe markers RTOEM / LTOEM correspond to the same anatomical placement as RMETA2 / LMETA2 in the
  older dataset: foot distal point = mean(heel, toe_meta) == mean(RHEEL, RTOEM), mean(LHEEL, LTOEM).

Data format:
- Header lines until MARKER_NAMES, then TRAJECTORY_TYPES, then rows of: Time + 3*N marker coords
  (no 'Frame' header line before data).

Output columns match the existing body_segments convention (Frame, Time, *_a_x/y/z, *_b_x/y/z).
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from typing import Dict, Iterable, List, Tuple

import numpy as np

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)

from s03_segment_qtm_raw import SEGMENTS

# Piano marker name -> canonical name used by compute_segments() in build_body_segments_from_qtm_tsv.py
PIANO_TO_CANONICAL: Dict[str, str] = {
    "HEADT": "HEAD_TOP",
    "HEADL": "HEAD_LEFT",
    "HEADR": "HEAD_RIGHT",
    "HEADA": "HEAD_FORHEAD",
    "ACROR": "RSHOULDER",
    "ACROL": "LSHOULDER",
    "ASISR": "RASIS",
    "ASISL": "LASIS",
    "PSISR": "RPSIS",
    "PSISL": "LPSIS",
    "HIPR": "RGT",
    "HIPL": "LGT",
    "RKNEEL": "RLKNEE",
    "RKNEEM": "RMKNEE",
    # LKNEEL -> LLKNEE/LMKNEE duplicate in remap_piano_markers()
    "RANKLEL": "RLANKLE",
    "RTOEL": "RMANKLE",
    "LANKLEL": "LLANKLE",
    "LTOEM": "LMANKLE",
    "RELBL": "RLELBOW",
    "RELBM": "RMELBOW",
    "LELBL": "LLELBOW",
    "LELBM": "LMELBOW",
    "RLAW": "RLWRIST",
    "RMEW": "RMWRIST",
    "LLAW": "LLWRIST",
    "LMEW": "LMWRIST",
    "RHAND": "RHAND",
    "LHAND": "LHAND",
    # Toe/meta placement == old RMETA2 / LMETA2
    "RTOEM": "RMETA2",
    "LTOEM": "LMETA2",
    "RHEEL": "RHEEL",
    "LHEEL": "LHEEL",
    "RUPARM1": "RUPPERARM1",
    "RUPARM2": "RUPPERARM2",
    "LUPARM1": "LUPPERARM1",
    "LUPARM2": "LUPPERARM2",
    "RLOWARM1": "RLOWERARM1",
    "RLOWARM2": "RLOWERARM2",
    "LLOWARM1": "LLOWERARM1",
    "LLOWARM2": "LLOWERARM2",
    "RTHIGH1": "RTHIGH1",
    "RTHIGH2": "RTHIGH2",
    "LTHIGH1": "LTHIGH1",
    "LTHIGH2": "LTHIGH2",
    "RSHANK1": "RSHANK1",
    "RSHANK2": "RSHANK2",
    "LSHANK1": "LSHANK1",
    "LSHANK2": "LSHANK2",
}


def remap_piano_markers(
    raw: Dict[str, np.ndarray],
) -> Dict[str, np.ndarray]:
    """Piano names -> canonical; duplicate LKNEEL to LMKNEE."""
    m: Dict[str, np.ndarray] = {}
    for k, v in raw.items():
        canon = PIANO_TO_CANONICAL.get(k)
        if canon:
            m[canon] = v.copy()
    if "LKNEEL" in raw:
        m["LLKNEE"] = raw["LKNEEL"].copy()
        m["LMKNEE"] = raw["LKNEEL"].copy()
    return m


def iter_tsv_rows(path: str) -> Iterable[List[str]]:
    with open(path, "r", newline="", encoding="utf-8", errors="replace") as f:
        reader = csv.reader(f, delimiter="\t")
        for row in reader:
            yield row


def parse_piano_position_tsv(path: str) -> Iterable[Tuple[int, float, Dict[str, np.ndarray]]]:
    """
    Yields (frame_index, time_sec, raw_marker_dict).

    Some Qualisys exports have data rows as exactly 3*N numbers (no leading Time column);
    others use one leading time column. If no time column, Time = (frame-1) / FREQUENCY.
    """
    marker_names: List[str] = []
    phase = "header"
    hz = 100.0
    frame_idx = 0

    for row in iter_tsv_rows(path):
        if not row:
            continue
        key = row[0].strip()

        if key == "FREQUENCY" and len(row) > 1:
            try:
                hz = float(row[1])
            except Exception:
                pass
            continue

        if key == "MARKER_NAMES":
            marker_names = [x.strip() for x in row[1:] if x.strip()]
            continue

        if key == "TRAJECTORY_TYPES":
            phase = "data"
            continue

        if phase != "data":
            continue

        n = len(marker_names)
        need = 3 * n
        if n == 0 or need <= 0:
            continue

        t: float
        cols: List[str]

        # Case A: only marker XYZ (common in this piano file: 54*3 = 162 cells)
        if len(row) == need:
            cols = row
            frame_idx += 1
            t = (frame_idx - 1) / hz
        # Case B: leading time column + marker XYZ
        elif len(row) == need + 1:
            try:
                t = float(row[0])
            except Exception:
                continue
            cols = row[1:]
            frame_idx += 1
        else:
            continue

        raw: Dict[str, np.ndarray] = {}
        for i, name in enumerate(marker_names):
            base = i * 3
            try:
                x, y, z = float(cols[base]), float(cols[base + 1]), float(cols[base + 2])
            except Exception:
                x = y = z = float("nan")
            raw[name] = np.array([x, y, z], dtype=float)

        yield frame_idx, t, raw


def compute_segments_piano(m: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    """Same topology as S5 compute_segments but hand without fingertip; meta2 from RTOEM/LTOEM."""
    from s03_segment_qtm_raw import _get, _mean

    out: Dict[str, np.ndarray] = {}

    # Visual3D head segment: proximal at the midpoint of the two lateral head
    # markers (cranial base / ear-line approximation), distal at the apex.
    # HEAD_FORHEAD defines the anterior axis but is NOT an endpoint — including
    # it in the proximal mean pulls head_a *inside* the skull.
    out["head_a"] = _mean(_get(m, "HEAD_LEFT"), _get(m, "HEAD_RIGHT"))
    out["head_b"] = _get(m, "HEAD_TOP")

    out["torso_a"] = _mean(_get(m, "RASIS"), _get(m, "LASIS"), _get(m, "RPSIS"), _get(m, "LPSIS"))
    out["torso_b"] = _mean(_get(m, "RSHOULDER"), _get(m, "LSHOULDER"))

    out["right_upper_arm_a"] = _get(m, "RSHOULDER")
    out["right_upper_arm_b"] = _get(m, "RLELBOW")
    out["right_forearm_a"] = _get(m, "RLELBOW")
    out["right_forearm_b"] = _get(m, "RLWRIST")
    out["right_hand_a"] = _get(m, "RLWRIST")
    out["right_hand_b"] = _get(m, "RHAND")

    out["left_upper_arm_a"] = _get(m, "LSHOULDER")
    out["left_upper_arm_b"] = _get(m, "LLELBOW")
    out["left_forearm_a"] = _get(m, "LLELBOW")
    out["left_forearm_b"] = _get(m, "LLWRIST")
    out["left_hand_a"] = _get(m, "LLWRIST")
    out["left_hand_b"] = _get(m, "LHAND")

    out["right_thigh_a"] = _mean(_get(m, "RASIS"), _get(m, "RGT"))
    out["right_thigh_b"] = _get(m, "RLKNEE")
    out["right_calf_a"] = _get(m, "RLKNEE")
    out["right_calf_b"] = _get(m, "RLANKLE")
    # Foot segment requires the toe marker (RMETA2 / LMETA2 in canonical;
    # RTOEM/RTOEF in the piano rig). Without it, _mean(HEEL, NaN) collapses to
    # HEEL alone — the stick then points backward from ankle to heel and looks
    # like a twisted ankle. When the toe is missing, drop the segment entirely
    # so the renderer doesn't draw a misleading line.
    if "RMETA2" in m:
        out["right_foot_a"] = _get(m, "RLANKLE")
        out["right_foot_b"] = _mean(_get(m, "RHEEL"), _get(m, "RMETA2"))
    else:
        out["right_foot_a"] = np.array([np.nan, np.nan, np.nan], dtype=float)
        out["right_foot_b"] = np.array([np.nan, np.nan, np.nan], dtype=float)

    out["left_thigh_a"] = _mean(_get(m, "LASIS"), _get(m, "LGT"))
    out["left_thigh_b"] = _get(m, "LLKNEE")
    out["left_calf_a"] = _get(m, "LLKNEE")
    out["left_calf_b"] = _get(m, "LLANKLE")
    if "LMETA2" in m:
        out["left_foot_a"] = _get(m, "LLANKLE")
        out["left_foot_b"] = _mean(_get(m, "LHEEL"), _get(m, "LMETA2"))
    else:
        out["left_foot_a"] = np.array([np.nan, np.nan, np.nan], dtype=float)
        out["left_foot_b"] = np.array([np.nan, np.nan, np.nan], dtype=float)

    return out


def write_body_segments_piano(in_path: str, out_path: str) -> None:
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

    fieldnames = ["Frame", "Time"]
    for seg in SEGMENTS:
        for end in ("a", "b"):
            for axis in ("x", "y", "z"):
                fieldnames.append(f"{seg}_{end}_{axis}")

    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t")
        w.writeheader()

        for frame_idx, t, raw in parse_piano_position_tsv(in_path):
            m = remap_piano_markers(raw)
            segs = compute_segments_piano(m)
            row: Dict[str, float] = {"Frame": frame_idx, "Time": t}
            for seg in SEGMENTS:
                for end in ("a", "b"):
                    p = segs[f"{seg}_{end}"]
                    row[f"{seg}_{end}_x"] = float(p[0])
                    row[f"{seg}_{end}_y"] = float(p[1])
                    row[f"{seg}_{end}_z"] = float(p[2])
            w.writerow(row)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("tsv", help="Input piano-style Qualisys TSV")
    ap.add_argument(
        "--out",
        default=None,
        help="Output path (default: <input>_body_segments.tsv next to input)",
    )
    args = ap.parse_args()

    in_path = args.tsv
    out_path = args.out
    if not out_path:
        base, _ = os.path.splitext(in_path)
        out_path = base + "_body_segments.tsv"
    write_body_segments_piano(in_path, out_path)
    print("WROTE", out_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
