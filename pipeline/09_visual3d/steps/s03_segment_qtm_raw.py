#!/usr/bin/env python3
"""
Generate body segment endpoint TSVs from Qualisys/QTM position raw TSV files.

This matches the existing '*_body_segments.tsv' convention found in this dataset:
- Each segment is represented by two 3D points: '<segment>_a_(x|y|z)' and '<segment>_b_(x|y|z)'.

Segments created (13):
  head, torso,
  right_upper_arm, right_forearm, right_hand,
  left_upper_arm,  left_forearm,  left_hand,
  right_thigh, right_calf, right_foot,
  left_thigh,  left_calf,  left_foot

Definitions (per frame). Head follows the Visual3D 4-marker convention:
proximal endpoint at the lateral midpoint, distal at the apex; HEAD_FORHEAD
is reserved for the anterior axis (not an endpoint).
  head_a = mean(HEAD_LEFT, HEAD_RIGHT)
  head_b = HEAD_TOP

  torso_a = mean(RASIS, LASIS, RPSIS, LPSIS)
  torso_b = mean(RSHOULDER, LSHOULDER)

  right_upper_arm_a = RSHOULDER
  right_upper_arm_b = RLELBOW
  right_forearm_a   = RLELBOW
  right_forearm_b   = RLWRIST
  right_hand_a      = RLWRIST
  right_hand_b      = mean(RHAND, RFINGERTIP)

  left_upper_arm_a = LSHOULDER
  left_upper_arm_b = LLELBOW
  left_forearm_a   = LLELBOW
  left_forearm_b   = LLWRIST
  left_hand_a      = LLWRIST
  left_hand_b      = mean(LHAND, LFINGERTIP)

  right_thigh_a = mean(RASIS, RGT)
  right_thigh_b = RMKNEE
  right_calf_a  = RMKNEE
  right_calf_b  = RLANKLE
  right_foot_a  = RLANKLE
  right_foot_b  = mean(RHEEL, RMETA2)

  left_thigh_a = mean(LASIS, LGT)
  left_thigh_b = LLKNEE
  left_calf_a  = LLKNEE
  left_calf_b  = LLANKLE
  left_foot_a  = LLANKLE
  left_foot_b  = mean(LHEEL, LMETA2)
"""

from __future__ import annotations

import argparse
import csv
import os
import re
from typing import Dict, Iterable, List, Tuple

import numpy as np


SEGMENTS = [
    "head",
    "torso",
    "right_upper_arm",
    "right_forearm",
    "right_hand",
    "left_upper_arm",
    "left_forearm",
    "left_hand",
    "right_thigh",
    "right_calf",
    "right_foot",
    "left_thigh",
    "left_calf",
    "left_foot",
]


def _iter_tsv_rows(path: str) -> Iterable[List[str]]:
    with open(path, "r", newline="") as f:
        reader = csv.reader(f, delimiter="\t")
        for row in reader:
            yield row


def _mean(*pts: np.ndarray) -> np.ndarray:
    a = np.vstack(pts).astype(float)
    return np.nanmean(a, axis=0)


def _get(m: Dict[str, np.ndarray], name: str) -> np.ndarray:
    v = m.get(name)
    if v is None:
        return np.array([np.nan, np.nan, np.nan], dtype=float)
    return v


def parse_qtm_position_tsv(path: str) -> Iterable[Tuple[int, float, Dict[str, np.ndarray]]]:
    """
    Yields (frame, time, marker_dict) for each data row.
    """
    marker_names: List[str] = []
    in_data = False

    for row in _iter_tsv_rows(path):
        if not row:
            continue
        key = row[0].strip()

        if key == "MARKER_NAMES":
            marker_names = [x for x in row[1:] if x.strip() != ""]
            continue

        if key == "Frame":
            in_data = True
            continue

        if not in_data:
            continue

        if len(row) < 2:
            continue
        try:
            frame = int(row[0])
            t = float(row[1])
        except Exception:
            continue

        cols = row[2:]
        m: Dict[str, np.ndarray] = {}
        for i, name in enumerate(marker_names):
            base = i * 3
            if base + 2 >= len(cols):
                break
            try:
                x = float(cols[base])
                y = float(cols[base + 1])
                z = float(cols[base + 2])
            except Exception:
                x = y = z = float("nan")
            m[name] = np.array([x, y, z], dtype=float)

        yield frame, t, m


def compute_segments(m: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    out: Dict[str, np.ndarray] = {}

    # Head — Visual3D segment: proximal at midpoint of LEFT/RIGHT lateral head
    # markers (cranial base / ear-line), distal at apex. HEAD_FORHEAD defines
    # the anterior axis but is not an endpoint.
    out["head_a"] = _mean(_get(m, "HEAD_LEFT"), _get(m, "HEAD_RIGHT"))
    out["head_b"] = _get(m, "HEAD_TOP")

    # Torso
    out["torso_a"] = _mean(_get(m, "RASIS"), _get(m, "LASIS"), _get(m, "RPSIS"), _get(m, "LPSIS"))
    out["torso_b"] = _mean(_get(m, "RSHOULDER"), _get(m, "LSHOULDER"))

    # Right arm
    out["right_upper_arm_a"] = _get(m, "RSHOULDER")
    out["right_upper_arm_b"] = _get(m, "RLELBOW")
    out["right_forearm_a"] = _get(m, "RLELBOW")
    out["right_forearm_b"] = _get(m, "RLWRIST")
    out["right_hand_a"] = _get(m, "RLWRIST")
    out["right_hand_b"] = _mean(_get(m, "RHAND"), _get(m, "RFINGERTIP"))

    # Left arm
    out["left_upper_arm_a"] = _get(m, "LSHOULDER")
    out["left_upper_arm_b"] = _get(m, "LLELBOW")
    out["left_forearm_a"] = _get(m, "LLELBOW")
    out["left_forearm_b"] = _get(m, "LLWRIST")
    out["left_hand_a"] = _get(m, "LLWRIST")
    out["left_hand_b"] = _mean(_get(m, "LHAND"), _get(m, "LFINGERTIP"))

    # Right leg
    out["right_thigh_a"] = _mean(_get(m, "RASIS"), _get(m, "RGT"))
    # In these files RMKNEE is often missing (0,0,0); existing body_segments use RLKNEE.
    out["right_thigh_b"] = _get(m, "RLKNEE")
    out["right_calf_a"] = _get(m, "RLKNEE")
    out["right_calf_b"] = _get(m, "RLANKLE")
    out["right_foot_a"] = _get(m, "RLANKLE")
    out["right_foot_b"] = _mean(_get(m, "RHEEL"), _get(m, "RMETA2"))

    # Left leg
    out["left_thigh_a"] = _mean(_get(m, "LASIS"), _get(m, "LGT"))
    out["left_thigh_b"] = _get(m, "LLKNEE")
    out["left_calf_a"] = _get(m, "LLKNEE")
    out["left_calf_b"] = _get(m, "LLANKLE")
    out["left_foot_a"] = _get(m, "LLANKLE")
    out["left_foot_b"] = _mean(_get(m, "LHEEL"), _get(m, "LMETA2"))

    return out


def write_body_segments(in_path: str, out_path: str) -> None:
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    fieldnames = ["Frame", "Time"]
    for seg in SEGMENTS:
        for end in ("a", "b"):
            for axis in ("x", "y", "z"):
                fieldnames.append(f"{seg}_{end}_{axis}")

    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t")
        w.writeheader()

        for frame, t, m in parse_qtm_position_tsv(in_path):
            segs = compute_segments(m)
            row: Dict[str, float] = {"Frame": frame, "Time": t}
            for seg in SEGMENTS:
                for end in ("a", "b"):
                    p = segs[f"{seg}_{end}"]
                    row[f"{seg}_{end}_x"] = float(p[0])
                    row[f"{seg}_{end}_y"] = float(p[1])
                    row[f"{seg}_{end}_z"] = float(p[2])
            w.writerow(row)


RAW_RE = re.compile(r"^S\d+_\d{4}\.tsv$", re.IGNORECASE)


def looks_like_raw_position_tsv(path: str) -> bool:
    # Quick sniff: needs MARKER_NAMES and Frame header
    try:
        with open(path, "r", newline="") as f:
            for _ in range(80):
                line = f.readline()
                if not line:
                    break
                if line.startswith("MARKER_NAMES\t"):
                    return True
    except Exception:
        return False
    return False


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=None, help="Root folder containing subject dirs (e.g. '2025 position and force data')")
    ap.add_argument("--subject", default=None, help="Only process one subject dir name (e.g. S5)")
    ap.add_argument("--in", dest="in_file", default=None, help="Process one specific input TSV file")
    ap.add_argument("--out-suffix", default="_body_segments_generated.tsv", help="Output suffix to append (default keeps originals intact)")
    args = ap.parse_args()

    if args.in_file:
        in_path = args.in_file
        out_path = os.path.splitext(in_path)[0] + args.out_suffix
        write_body_segments(in_path, out_path)
        print("WROTE", out_path)
        return 0

    if not args.root:
        raise SystemExit("--root is required unless --in is provided")

    root = args.root
    subjects = []
    for name in sorted(os.listdir(root)):
        p = os.path.join(root, name)
        if not os.path.isdir(p):
            continue
        if args.subject and name != args.subject:
            continue
        if re.match(r"^S\d+$", name, re.IGNORECASE):
            subjects.append(p)

    wrote = 0
    for subj_dir in subjects:
        for fn in sorted(os.listdir(subj_dir)):
            if not RAW_RE.match(fn):
                continue
            in_path = os.path.join(subj_dir, fn)
            if not looks_like_raw_position_tsv(in_path):
                continue
            out_path = os.path.splitext(in_path)[0] + args.out_suffix
            write_body_segments(in_path, out_path)
            wrote += 1

    print("WROTE_FILES", wrote)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

