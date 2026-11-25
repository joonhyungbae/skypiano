"""
Stage 07 Step 06: held-out comparison of the gap-filling methods.

Same protocol as s05: the held-out pieces of s05 (read from Eval/_errors.csv), cleaned by s00; observed
stretches of every marker are hidden on purpose (0.1 / 0.5 / 1 / 2 s, up to 3 gaps per marker, all
markers hidden in the same pass) and filled by each method of src/fill.py; error = mean 3D distance (mm)
to the hidden truth. Optional extra methods are read from Eval/extra/{method}/{piece}_{gap}.npz
(arrays `pred` = frames x 141, written by an external imputer on the same hidden input saved under
Eval/hidden/).

Output: Data/Pipeline/07_body_imputation/Eval/compare_errors.csv, compare_summary.csv
"""
import functools
import sys
import zlib
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.fill import fill
from src.utils import BODY_MARKERS, body_joint_cols

print = functools.partial(print, flush=True)

EVAL_ROOT = Path("Data/Pipeline/07_body_imputation/Eval")
CLEAN_ROOT = Path("Data/Pipeline/07_body_imputation/Clean")
GAPS = {"0.1s": 12, "0.5s": 60, "1s": 120, "2s": 240}
PER_MARKER = 3
METHODS = ["linear", "spline", "pca", "rigid", "rigid_pca", "rule"]
GROUPS = {"head": ["HEAD"], "trunk": ["ACRO", "TRUNK"], "pelvis": ["ASIS", "PSIS", "HIP"],
          "arm": ["UPARM", "LOWARM", "ELBL"], "leg": ["THIGH", "SHANK", "KNEE"],
          "foot": ["HEEL", "ANKLE", "TOE"]}


def group(m):
    return next((g for g, keys in GROUPS.items() for k in keys if k in m), "other")


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
    pid, piece, methods = job
    df = pd.read_csv(CLEAN_ROOT / piece[:6] / f"{piece}_body.csv")
    rows = []
    for gname, L in GAPS.items():
        rng = np.random.default_rng(zlib.crc32(f"{piece}|{gname}".encode()))
        hid, plan = df.copy(), []
        for m in BODY_MARKERS:
            if f"{m}_x" not in df.columns:
                continue
            for s in place_gaps(df[f"{m}_x"].notna().to_numpy(), L, rng):
                plan.append((m, s))
                hid.loc[s:s + L - 1, [f"{m}_x", f"{m}_y", f"{m}_z"]] = np.nan
        if not plan:
            continue
        hp = EVAL_ROOT / "hidden" / f"{piece}_{gname}.csv.gz"
        hp.parent.mkdir(parents=True, exist_ok=True)
        hid[[c for c in body_joint_cols if c in hid.columns]].to_csv(hp, index=False)
        marks = sorted({m for m, _ in plan})
        preds = {meth: fill(hid, meth, marks) for meth in methods}
        for meth in sorted(p.name for p in (EVAL_ROOT / "extra").glob("*")) if (EVAL_ROOT / "extra").exists() else []:
            f = EVAL_ROOT / "extra" / meth / f"{piece}_{gname}.npz"
            if f.exists():
                preds[meth] = pd.DataFrame(np.load(f)["pred"], columns=body_joint_cols)
        for m, s in plan:
            cols = [f"{m}_x", f"{m}_y", f"{m}_z"]
            true = df[cols].to_numpy()[s:s + L]
            row = {"pid": pid, "piece": piece, "gap": gname, "marker": m, "group": group(m)}
            for meth, p in preds.items():
                row[meth] = float(np.linalg.norm(p[cols].to_numpy()[s:s + L] - true, axis=1).mean())
            rows.append(row)
    print(f"  {piece}: {len(rows)} gaps")
    return rows


def run(workers, methods):
    split = pd.read_csv(EVAL_ROOT / "_errors.csv")[["pid", "piece"]].drop_duplicates()
    jobs = [(r.pid, r.piece, methods) for r in split.itertuples()]
    print(f"Stage 07 s06: {len(jobs)} held-out pieces, methods {methods}, workers {workers}")
    with ProcessPoolExecutor(workers) as ex:
        rows = [r for part in ex.map(one, jobs) for r in part]
    e = pd.DataFrame(rows)
    e.to_csv(EVAL_ROOT / "compare_errors.csv", index=False)
    cols = [c for c in e.columns if c not in ("pid", "piece", "gap", "marker", "group")]
    order = list(GAPS)
    s = e.groupby("gap")[cols].mean().round(2).loc[order]
    s.to_csv(EVAL_ROOT / "compare_summary.csv")
    print("\nmean 3D error (mm) by gap length\n" + s.to_string())
    print("\nmedian\n" + e.groupby("gap")[cols].median().round(2).loc[order].to_string())
    print("\nby body part, 1 s gaps\n" + e[e.gap == "1s"].groupby("group")[cols].mean().round(2).to_string())
    print("\nby performer, 1 s gaps\n" + e[e.gap == "1s"].groupby("pid")[cols].mean().round(2).to_string())


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--methods", nargs="*", default=METHODS)
    a = ap.parse_args()
    run(a.workers, a.methods)
