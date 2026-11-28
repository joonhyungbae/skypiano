#!/usr/bin/env python3
"""
Stage 09 step s06: Visual3D-style body joints from a body marker CSV, vectorised and strict.

Same segment topology as s02.compute_segments_piano (13 segments, each an a/b endpoint pair), but a point
that averages several markers is NaN when any of them is missing: a partial mean would silently move the
point (e.g. the pelvis centre by centimetres when one ASIS/PSIS marker drops out), so the unfilled release
keeps such frames as gaps instead of biased values.

Unique points (mm, release frame of the input CSV):
  head_top HEADT, head_base mean(HEADR, HEADL), head_front HEADA (anterior reference only),
  neck mean(ACROR, ACROL), pelvis mean(ASISR, ASISL, PSISR, PSISL),
  {r,l}_shoulder ACRO, {r,l}_elbow ELBL, {r,l}_wrist LAW, {r,l}_hand HAND (220819 / 221112 only),
  {r,l}_hip mean(ASIS, HIP), {r,l}_knee KNEEL, {r,l}_ankle ANKLEL, {r,l}_foot mean(HEEL, TOEM | TOEF).

Usage:
  python s06_joints.py --in <piece>_body.csv --out <piece>_v3d.csv
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

POINTS = {
    "head_top": ["HEADT"], "head_base": ["HEADR", "HEADL"], "head_front": ["HEADA"],
    "neck": ["ACROR", "ACROL"], "pelvis": ["ASISR", "ASISL", "PSISR", "PSISL"],
}
for _s, _S in (("r", "R"), ("l", "L")):
    POINTS.update({
        f"{_s}_shoulder": [f"ACRO{_S}"], f"{_s}_elbow": [f"{_S}ELBL"], f"{_s}_wrist": [f"{_S}LAW"],
        f"{_s}_hand": [f"{_S}HAND"], f"{_s}_hip": [f"ASIS{_S}", f"HIP{_S}"], f"{_s}_knee": [f"{_S}KNEEL"],
        f"{_s}_ankle": [f"{_S}ANKLEL"], f"{_s}_foot": [f"{_S}HEEL", f"{_S}TOE*"],
    })
COLUMNS = [f"{p}_{a}" for p in POINTS for a in "xyz"]


def _marker(df: pd.DataFrame, m: str) -> np.ndarray:
    if m.endswith("*"):                                   # toe: TOEM, else TOEF (meta-2 placement)
        for alt in (m[:-1] + "M", m[:-1] + "F"):
            if f"{alt}_x" in df.columns:
                return _marker(df, alt)
        m = m[:-1] + "M"
    cols = [f"{m}_{a}" for a in "xyz"]
    if cols[0] not in df.columns:
        return np.full((len(df), 3), np.nan)
    return df[cols].to_numpy(dtype=float)


def joints(df: pd.DataFrame) -> pd.DataFrame:
    """Marker frame (one row per frame, <MARKER>_x|y|z in mm) -> joint frame (COLUMNS); strict means."""
    out = {}
    for p, ms in POINTS.items():
        v = np.mean(np.stack([_marker(df, m) for m in ms]), axis=0)   # NaN if any marker is NaN
        for k, a in enumerate("xyz"):
            out[f"{p}_{a}"] = v[:, k]
    return pd.DataFrame(out, columns=COLUMNS)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    joints(pd.read_csv(a.inp)).to_csv(a.out, index=False, float_format="%.3f")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
