"""
Stage 07 Step 04: replace NaN in original body CSV with imputed values.

Input:
  - Data/Pipeline/04_piece_split/{dataset}/{piece_prefix}_body.csv (original)
  - Data/Pipeline/07_body_imputation/Imp/result/{pid}/{piece_prefix}.csv (imputed)
Output:
  - Data/Pipeline/07_body_imputation/Final/{pid}/{piece_prefix}.csv
    (body markers only; wrist markers RLAW/LLAW dropped — those come from Stage 06)
"""
import functools
import sys
from pathlib import Path

import pandas as pd
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.utils import body_joint_cols, BODY_EXCLUDED
from src.performers import PERFORMER_MAP, performer_id, get_piece_body_path

print = functools.partial(print, flush=True)

IMP_ROOT = Path("Data/Pipeline/07_body_imputation/Imp/result")
OUT_ROOT = Path("Data/Pipeline/07_body_imputation/Final")

EXCLUDED_COLS = {f"{m}_{ax}" for m in BODY_EXCLUDED for ax in ("x", "y", "z")}


def run(only=None, skip=None):
    print("Stage 07 s04: Replace NaN with imputed values (drop wrist markers)")
    if only: print(f"  only: {only}")
    if skip: print(f"  skip: {skip}")
    print(f"  dropping from output: {sorted(EXCLUDED_COLS)}")

    for (c, n), pieces in sorted(PERFORMER_MAP.items()):
        pid = performer_id(c, n)
        if only and pid not in only: continue
        if skip and pid in skip:
            print(f"  {pid}: skip (excluded)")
            continue

        imp_dir = IMP_ROOT / pid
        out_dir = OUT_ROOT / pid
        if not imp_dir.exists():
            continue

        remaining = [(ds, pp) for ds, pp in pieces
                     if not (out_dir / f"{pp}.csv").exists()
                     and (imp_dir / f"{pp}.csv").exists()]
        if not remaining:
            print(f"  {pid}: skip")
            continue

        out_dir.mkdir(parents=True, exist_ok=True)
        print(f"  {pid}: {len(remaining)} files")
        for dataset, piece_prefix in tqdm(remaining, desc=f"    {pid}"):
            data = pd.read_csv(get_piece_body_path(dataset, piece_prefix))
            imp = pd.read_csv(imp_dir / f"{piece_prefix}.csv")

            available = [col for col in body_joint_cols
                         if col in imp.columns and col in data.columns]
            for col in available:
                mask = data[col].isna()
                if mask.any():
                    data.loc[mask, col] = imp.loc[mask, col]

            keep_cols = [col for col in data.columns if col not in EXCLUDED_COLS]
            data[keep_cols].to_csv(str(out_dir / f"{piece_prefix}.csv"), index=False)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*")
    ap.add_argument("--skip", nargs="*")
    args = ap.parse_args()
    run(only=args.only, skip=args.skip)
