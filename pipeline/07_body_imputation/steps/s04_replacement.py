"""
Stage 07 Step 04: fill the gaps of the cleaned body CSV.

Each gap (NaN run with an observed frame on both sides) of at most MAX_GAP_SEC is filled by the `rule`
method of src/fill.py (training-free; per body part and gap length, the most accurate method of the
held-out comparison s06). Longer gaps, gaps touching the start or end of the piece, and markers the piece
never observes stay empty (mask 0). Wrist markers RLAW/LLAW are never filled (they come from Stage 06).

Input:
  - Data/Pipeline/07_body_imputation/Clean/{dataset}/{piece_prefix}_body.csv     (s00)
  - Data/Pipeline/07_body_imputation/Clean/{dataset}/{piece_prefix}_outlier.csv  (s00, 1 = removed)
Output:
  - Data/Pipeline/07_body_imputation/Final/{pid}/{piece_prefix}.csv
    (body markers only; wrist markers RLAW/LLAW dropped; input of Stage 09)
  - Data/Pipeline/07_body_imputation/Release/{dataset}/{piece_prefix}_body_imputed.csv
    (same columns and rows as the released body.csv; RLAW/LLAW kept as cleaned, never filled)
  - .../{piece_prefix}_body_imputed_mask.csv   (1 = value filled here)
  - .../{piece_prefix}_body_suspect_mask.csv   (1 = released body.csv value removed as an outlier by s00)
"""
import functools
import shutil
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.fill import FPS, fill
from src.utils import BODY_EXCLUDED, BODY_MARKERS
from src.performers import PERFORMER_MAP, performer_id, get_clean_body_path, CLEAN_ROOT

print = functools.partial(print, flush=True)

OUT_ROOT = Path("Data/Pipeline/07_body_imputation/Final")
REL_ROOT = Path("Data/Pipeline/07_body_imputation/Release")
MAX_GAP_SEC = 2.0

EXCLUDED_COLS = {f"{m}_{ax}" for m in BODY_EXCLUDED for ax in ("x", "y", "z")}


def one(job):
    pid, ds, pp = job
    data = pd.read_csv(get_clean_body_path(ds, pp))
    markers = [m for m in BODY_MARKERS if f"{m}_x" in data.columns]
    out = fill(data, "rule", markers, max_gap=int(round(MAX_GAP_SEC * FPS)))
    mask = (data.isna() & out.notna()).astype(np.int8)
    (OUT_ROOT / pid).mkdir(parents=True, exist_ok=True)
    out[[c for c in out.columns if c not in EXCLUDED_COLS]].to_csv(OUT_ROOT / pid / f"{pp}.csv", index=False)
    rel = REL_ROOT / ds
    rel.mkdir(parents=True, exist_ok=True)
    out.to_csv(rel / f"{pp}_body_imputed.csv", index=False)
    mask.to_csv(rel / f"{pp}_body_imputed_mask.csv", index=False)
    shutil.copyfile(CLEAN_ROOT / ds / f"{pp}_outlier.csv", rel / f"{pp}_body_suspect_mask.csv")
    return pid, int(mask.to_numpy().sum()), int(data.isna().to_numpy().sum()), mask.size


def run(only=None, skip=None, force=False, workers=8):
    print(f"Stage 07 s04: fill gaps up to {MAX_GAP_SEC} s (rule)")
    jobs = []
    for (c, n), pieces in sorted(PERFORMER_MAP.items()):
        pid = performer_id(c, n)
        if (only and pid not in only) or (skip and pid in skip):
            continue
        jobs += [(pid, ds, pp) for ds, pp in pieces if get_clean_body_path(ds, pp).exists()
                 and (force or not (REL_ROOT / ds / f"{pp}_body_imputed.csv").exists())]
    print(f"  {len(jobs)} pieces, workers {workers}")
    tot = {}
    with ProcessPoolExecutor(workers) as ex:
        for k, (pid, filled, missing, cells) in enumerate(ex.map(one, jobs), 1):
            t = tot.setdefault(pid, [0, 0, 0, 0])
            t[0] += 1; t[1] += filled; t[2] += missing; t[3] += cells
            if k % 25 == 0:
                print(f"  {k}/{len(jobs)}")
    for pid, (n, filled, missing, cells) in sorted(tot.items()):
        print(f"  {pid}: {n} pieces, {filled / max(cells, 1) * 100:.2f}% of cells filled, "
              f"{(missing - filled) / max(cells, 1) * 100:.2f}% left empty")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*")
    ap.add_argument("--skip", nargs="*")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()
    run(only=args.only, skip=args.skip, force=args.force, workers=args.workers)
