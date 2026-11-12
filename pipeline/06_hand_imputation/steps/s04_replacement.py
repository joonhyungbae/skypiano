"""
Stage 06 Step 04: Per-performer first-pass imputation replacement (Stage 04 per-piece based)

Input:
  - Data/Pipeline/04_piece_split/{dataset}/{piece_prefix}_hand.csv (original)
  - Data/Pipeline/06_hand_imputation/Imp/result/Body/{performer_id}/{piece_prefix}.csv (imputed)
Output:
  - Data/Pipeline/06_hand_imputation/Imp/replace/Body/{performer_id}/{piece_prefix}.csv
"""
import functools
import sys
from pathlib import Path

import pandas as pd
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.utils import body_fill_cols
from src.performers import PERFORMER_MAP, performer_id, get_piece_hand_path

print = functools.partial(print, flush=True)

IMP_ROOT = Path("Data/Pipeline/06_hand_imputation/Imp/result/Body")
OUT_ROOT = Path("Data/Pipeline/06_hand_imputation/Imp/replace/Body")


def run(only: list[str] | None = None, skip: list[str] | None = None):
    print("Stage 06 s04: Replace NaN with imputed values per performer")
    if only:
        print(f"  only: {only}")
    if skip:
        print(f"  skip: {skip}")

    for (c, n), pieces in sorted(PERFORMER_MAP.items()):
        pid = performer_id(c, n)
        if only and pid not in only:
            continue
        if skip and pid in skip:
            print(f"  {pid}: skip (excluded by flag)")
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
            data = pd.read_csv(get_piece_hand_path(dataset, piece_prefix))
            imp_data = pd.read_csv(imp_dir / f"{piece_prefix}.csv")

            available = [col for col in body_fill_cols
                         if col in imp_data.columns and col in data.columns]
            for col in available:
                mask = data[col].isna()
                if mask.any():
                    data.loc[mask, col] = imp_data.loc[mask, col]

            data.to_csv(str(out_dir / f"{piece_prefix}.csv"), index=False)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*", help="performer IDs to include")
    ap.add_argument("--skip", nargs="*", help="performer IDs to exclude")
    args = ap.parse_args()
    run(only=args.only, skip=args.skip)
