#!/usr/bin/env python3
"""
Compute per-frame segment center points from '*_body_segments*.tsv' files.

For each segment, center = (a + b) / 2 for x/y/z.

Input columns expected:
  <segment>_a_x, <segment>_a_y, <segment>_a_z,
  <segment>_b_x, <segment>_b_y, <segment>_b_z

Output columns:
  <segment>_c_x, <segment>_c_y, <segment>_c_z

Default output name:
- If input ends with '_body_segments.tsv' → replace with '_body_segment_centers.tsv'
- Otherwise → append '_body_segment_centers.tsv'
"""

from __future__ import annotations

import argparse
import csv
import os
import re
from typing import Dict, Iterable, List


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


def iter_rows(path: str) -> Iterable[Dict[str, str]]:
    # Some dataset files may not be strict UTF-8; numbers are still readable.
    with open(path, "r", newline="", encoding="utf-8", errors="replace") as f:
        r = csv.DictReader(f, delimiter="\t")
        for row in r:
            yield row


def fnum(x: str) -> float:
    try:
        return float(x)
    except Exception:
        return float("nan")


def write_centers(in_path: str, out_path: str) -> None:
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    fieldnames = ["Frame", "Time"]
    for seg in SEGMENTS:
        for axis in ("x", "y", "z"):
            fieldnames.append(f"{seg}_c_{axis}")

    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t")
        w.writeheader()

        for row in iter_rows(in_path):
            out: Dict[str, float] = {"Frame": int(float(row["Frame"])), "Time": float(row["Time"])}
            for seg in SEGMENTS:
                ax = fnum(row.get(f"{seg}_a_x", "nan"))
                ay = fnum(row.get(f"{seg}_a_y", "nan"))
                az = fnum(row.get(f"{seg}_a_z", "nan"))
                bx = fnum(row.get(f"{seg}_b_x", "nan"))
                by = fnum(row.get(f"{seg}_b_y", "nan"))
                bz = fnum(row.get(f"{seg}_b_z", "nan"))
                out[f"{seg}_c_x"] = (ax + bx) / 2.0
                out[f"{seg}_c_y"] = (ay + by) / 2.0
                out[f"{seg}_c_z"] = (az + bz) / 2.0
            w.writerow(out)


_REPLACE_BODY_SEGMENTS = re.compile(r"(?i)_body_segments[^/]*\.tsv$")
_MATCH_BODY_SEGMENTS = re.compile(r"(?i)_body_segments[^/]*\.tsv$")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=None, help="Root folder containing subject dirs (e.g. '2025 position and force data')")
    ap.add_argument("--in", dest="in_file", default=None, help="Process one specific input TSV file")
    ap.add_argument("--out", default=None, help="Explicit output path (.tsv)")
    ap.add_argument("--out-suffix", default="_body_segment_centers.tsv", help="Used only with --root mode")
    args = ap.parse_args()

    if args.in_file:
        in_path = args.in_file
        if args.out:
            out_path = args.out
        else:
            if _REPLACE_BODY_SEGMENTS.search(in_path):
                # Preserve any extra suffix after '_body_segments' (e.g. '_ext')
                out_path = re.sub(
                    r"(?i)_body_segments",
                    "_body_segment_centers",
                    in_path,
                    count=1,
                )
            else:
                out_path = in_path + "_body_segment_centers.tsv"
        write_centers(in_path, out_path)
        print("WROTE", out_path)
        return 0

    if not args.root:
        raise SystemExit("--root is required unless --in is provided")

    wrote = 0
    for dirpath, _, filenames in os.walk(args.root):
        for fn in filenames:
            if not _MATCH_BODY_SEGMENTS.search(fn):
                continue
            in_path = os.path.join(dirpath, fn)
            out_path = _REPLACE_BODY_SEGMENTS.sub(args.out_suffix, in_path)
            write_centers(in_path, out_path)
            wrote += 1

    print("WROTE_FILES", wrote)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

