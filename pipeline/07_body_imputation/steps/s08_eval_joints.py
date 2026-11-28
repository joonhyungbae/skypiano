"""
Stage 07 Step 08: held-out accuracy of the gap-filled Visual3D joints (Stage 09 s06).

Same held-out pieces and gap placement as s06 (all body markers, the wrist and the hand markers hidden in
one pass, up to 3 gaps per marker), extended to 3 s and 5 s gaps. Truth = joints of the unhidden piece.
Scored on every joint sample that the hiding removed (NaN in the hidden joints, valid in the truth).

Methods (m_rule_h<w>: rule with the hand-capture arm markers as extra donors, Kabsch weight w):
  m_<method>  fill the markers with src/fill.py <method>, then compute the joints.
  j_<method>  compute the joints of the hidden markers (strict: NaN when a defining marker is missing),
              then fill each joint trajectory with <method> (linear | spline).

Output: Data/Pipeline/07_body_imputation/Eval/joints_errors.csv (one row per piece x gap x joint)
"""
import functools
import sys
import zlib
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "07_body_imputation"))
sys.path.insert(0, str(ROOT / "09_visual3d" / "steps"))
import src.fill as F
from src.fill import fill
from src.utils import BODY_MARKERS
from s06_joints import POINTS, joints

print = functools.partial(print, flush=True)

EVAL = Path("Data/Pipeline/07_body_imputation/Eval")
CLEAN = Path("Data/Pipeline/07_body_imputation/Clean")
HAND = Path("Data/Pipeline/04_piece_split")
HAND_WEIGHTS = {"rule_h01": 0.1, "rule_h03": 0.3, "rule_h1": 1.0}
GAPS = {"0.1s": 12, "0.5s": 60, "1s": 120, "2s": 240, "3s": 360, "5s": 600}
PER_MARKER = 3
EXTRA = ["RLAW", "LLAW", "RHAND", "LHAND"]
MARKER_METHODS = ["linear", "spline", "rigid", "rule"]
JOINT_METHODS = ["linear", "spline"]


def place_gaps(obs, length, rng):
    ok = np.convolve(obs.astype(int), np.ones(length + 2, int), "valid") == length + 2
    starts, taken = [], np.zeros(len(obs), bool)
    for s in rng.permutation(np.flatnonzero(ok)):
        if len(starts) == PER_MARKER:
            break
        if not taken[s:s + length + 2].any():
            starts.append(s + 1); taken[s:s + length + 2] = True
    return starts


def one(job):
    pid, piece, path = job
    df = pd.read_csv(path)
    markers = [m for m in BODY_MARKERS + EXTRA if f"{m}_x" in df.columns]
    truth = joints(df)
    rows = []
    for gname, L in GAPS.items():
        rng = np.random.default_rng(zlib.crc32(f"{piece}|{gname}|joints".encode()))
        hid = df.copy()
        for m in markers:
            for s in place_gaps(df[f"{m}_x"].notna().to_numpy(), L, rng):
                hid.loc[s:s + L - 1, [f"{m}_x", f"{m}_y", f"{m}_z"]] = np.nan
        hj = joints(hid)
        preds = {f"m_{k}": joints(fill(hid, k, markers)) for k in MARKER_METHODS}
        preds.update({f"j_{k}": fill(hj, k, list(POINTS)) for k in JOINT_METHODS})
        hp = HAND / piece[:6] / f"{piece}_hand.csv"
        if hp.exists():
            h = pd.read_csv(hp)
            cols = [f"{m}_{a}" for m in sorted(F.HAND_DONORS) for a in "xyz" if f"{m}_x" in h.columns]
            hh = pd.concat([hid, h[cols].iloc[:len(hid)].reset_index(drop=True)], axis=1)
            for name, w in HAND_WEIGHTS.items():
                F.HAND_W = w
                preds[f"m_{name}"] = joints(fill(hh, "rule", markers))
        for p in POINTS:
            cols = [f"{p}_{a}" for a in "xyz"]
            sel = hj[cols[0]].isna().to_numpy() & truth[cols[0]].notna().to_numpy()
            if not sel.any():
                continue
            t = truth[cols].to_numpy()[sel]
            row = {"pid": pid, "piece": piece, "gap": gname, "joint": p, "n": int(sel.sum())}
            for k, pr in preds.items():
                e = np.linalg.norm(pr[cols].to_numpy()[sel] - t, axis=1)
                row[k] = float(np.nanmean(e)) if np.isfinite(e).any() else np.nan
                row[k + "_cov"] = float(np.isfinite(e).mean())
            rows.append(row)
    print(f"  {piece}: {len(rows)} rows")
    return rows


def run(workers):
    split = pd.read_csv(EVAL / "_errors.csv")[["pid", "piece"]].drop_duplicates()
    jobs = [(r.pid, r.piece, CLEAN / r.piece[:6] / f"{r.piece}_body.csv") for r in split.itertuples()]
    print(f"Stage 07 s08: {len(jobs)} held-out pieces, joints {len(POINTS)}")
    with ProcessPoolExecutor(workers) as ex:
        rows = [r for part in ex.map(one, jobs) for r in part]
    e = pd.DataFrame(rows)
    e.to_csv(EVAL / "joints_errors.csv", index=False)
    meth = [f"m_{k}" for k in MARKER_METHODS] + [f"m_{k}" for k in HAND_WEIGHTS] + [f"j_{k}" for k in JOINT_METHODS]

    def wmean(g):
        return pd.Series({k: np.average(g[k].fillna(0), weights=g["n"] * g[k].notna()) if g[k].notna().any()
                          else np.nan for k in meth})
    order = list(GAPS)
    print("\nmean 3D joint error (mm), weighted by samples\n" + e.groupby("gap").apply(wmean).round(2).loc[order].to_string())
    print("\ncoverage (share of removed samples filled)\n" +
          e.groupby("gap")[[k + "_cov" for k in meth]].mean().round(3).loc[order].to_string())
    print("\nby joint, 1 s gaps\n" + e[e.gap == "1s"].groupby("joint").apply(wmean).round(2).to_string())


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=8)
    run(ap.parse_args().workers)
