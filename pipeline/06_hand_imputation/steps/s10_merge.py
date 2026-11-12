"""
Stage 06 step 10: copy per-piece second-pass imputation results into the
training-input layout consumed by s08.

The upstream step saves piece-level concatenated CSVs under Imp/result/{L,R}/{pid}/.
This step copies those files as-is to Imp/concat/{L,R}/{pid}/, which is the
input path expected by the next training step.
"""
import os
import shutil
from tqdm import tqdm

from src.performers import PERFORMER_MAP, performer_id
from src.utils import makeMotionFilePathLst

MOTION_EXT = '.csv'
DIRECTIONS = ['Left', 'Right']
SRC_ROOT = 'Data/Pipeline/06_hand_imputation/Imp/result'
DST_ROOT = 'Data/Pipeline/06_hand_imputation/Imp/concat'


def main():
    for (c_val, n_val), _ in sorted(PERFORMER_MAP.items()):
        pid = performer_id(c_val, n_val)

        for direction in DIRECTIONS:
            src_dir = f'{SRC_ROOT}/{direction}/{pid}'
            dst_dir = f'{DST_ROOT}/{direction}/{pid}'

            if not os.path.isdir(src_dir):
                print(f"Skip {pid}/{direction}: no source dir")
                continue

            paths = makeMotionFilePathLst(src_dir, MOTION_EXT)
            if not paths:
                print(f"Skip {pid}/{direction}: no CSVs")
                continue

            os.makedirs(dst_dir, exist_ok=True)
            for src_path in tqdm(paths, desc=f"{pid}/{direction}"):
                filename = os.path.basename(src_path)
                shutil.copyfile(src_path, os.path.join(dst_dir, filename))


if __name__ == "__main__":
    main()
