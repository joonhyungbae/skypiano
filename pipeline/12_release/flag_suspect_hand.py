"""
Stage 12: flag implausible observations in the released hand motion.

The optical capture occasionally labels a ghost reflection or another marker as
one of the hand's markers: the value is recorded, but it is not where that
marker was. Such entries are kept in the released hand.csv unchanged and are
flagged here, so that users can drop them and the gap fill does not anchor on
them. An observed entry is flagged when

  far    it lies more than FAR_MM from the median of the same hand's observed
         markers in that frame (a hand spans about 200 mm; the arm and keyboard
         reference markers -- Sh, Elbow, End -- are not part of the hand), or
  spike  it moves more than SPIKE_MM in one frame into a position and more than
         SPIKE_MM out of it in the next (1/120 s each; 50 mm per frame is
         6 m/s) -- a one-frame excursion, not a reappearance after a gap.

Writes, per released performance (performances.csv `path`):
    Data/Pipeline/06_hand_imputation/Suspect/<release path>/hand_suspect_mask.csv
        same columns as hand.csv; 1 = observed but implausible, 0 otherwise
    Data/Pipeline/06_hand_imputation/Suspect/_summary.csv
        per performance and hand: entries flagged far / spike

Usage (from the repository root):
  python pipeline/12_release/flag_suspect_hand.py --version 0.1.0
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import pandas as pd

REL = Path("Data/Release")
OUT = Path("Data/Pipeline/06_hand_imputation/Suspect")
FAR_MM = 150.0
SPIKE_MM = 50.0
NOT_HAND = ("Sh", "Elbow", "End")


def suspect_mask(raw: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Mask of implausible observed entries (columns of `raw`), and counts per hand."""
    mask = pd.DataFrame(0, index=raw.index, columns=raw.columns, dtype=np.int8)
    counts = {}
    for side in ("L", "R"):
        markers = sorted({c[:-2] for c in raw.columns if c.startswith(side) and c.endswith("_x")
                          and not any(k in c for k in NOT_HAND)})
        if not markers:
            continue
        P = np.stack([raw[[m + "_x", m + "_y", m + "_z"]].to_numpy(float) for m in markers], 1)   # T x M x 3
        obs = ~np.isnan(P[..., 0])
        centre = np.nanmedian(P, axis=1, keepdims=True) if obs.any() else P[:, :1]
        far = (np.linalg.norm(P - centre, axis=2) > FAR_MM) & obs
        step = np.linalg.norm(np.diff(P, axis=0), axis=2)                        # T-1 x M, NaN across gaps
        spike = np.zeros_like(far)
        spike[1:-1] = (step[:-1] > SPIKE_MM) & (step[1:] > SPIKE_MM)
        bad = far | spike
        for j, m in enumerate(markers):
            for s in ("_x", "_y", "_z"):
                mask.loc[bad[:, j], m + s] = 1
        counts[side] = (int(far.sum()), int((spike & ~far).sum()))
    return mask, counts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--version", required=True)
    args = ap.parse_args()
    vdir = REL / f"v{args.version}"
    source = {m["release_path"]: m["source_path"] for m in csv.DictReader(open(vdir / "manifest.csv", encoding="utf-8"))}
    rows = []
    for perf in csv.DictReader(open(vdir / "performances.csv", encoding="utf-8")):
        src = source.get(f"{perf['path']}/hand.csv")
        if not src:
            continue
        raw = pd.read_csv(src)
        mask, counts = suspect_mask(raw)
        out = OUT / perf["path"]
        out.mkdir(parents=True, exist_ok=True)
        mask.to_csv(out / "hand_suspect_mask.csv", index=False)
        for side, (far, spike) in counts.items():
            rows.append({"path": perf["path"], "session": perf["session"], "hand": side, "far": far, "spike": spike})
    summary = pd.DataFrame(rows)
    summary.to_csv(OUT / "_summary.csv", index=False)
    flagged = summary[(summary.far + summary.spike) > 0]
    print(f"{summary.path.nunique()} performances; recording-hands with flags: {len(flagged)}; "
          f"entries flagged far {summary.far.sum()}, spike {summary.spike.sum()}")
    print(flagged.groupby("session")[["far", "spike"]].sum().to_string())


if __name__ == "__main__":
    main()
