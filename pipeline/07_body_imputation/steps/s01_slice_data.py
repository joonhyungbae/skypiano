"""
Stage 07 Step 01: slice per-performer body data into 3s windows with features.

Features per frame: raw + residual + velocity + acceleration (no connection-based
quaternion/distance — SAITS imputes directly from raw coords).

Input:  Data/Pipeline/04_piece_split/{dataset}/{piece_prefix}_body.csv
Output: Data/Pipeline/07_body_imputation/Sliced/{pid}/step1_sec3.npy
"""
import functools
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.utils import body_joint_cols
from src.performers import (
    PERFORMER_MAP,
    performer_id,
    get_piece_body_path,
    log_known_body_absences,
)

print = functools.partial(print, flush=True)

OUT_ROOT = Path("Data/Pipeline/07_body_imputation/Sliced")

FPS = 120
SEC = 3
STEP = 1
WINDOW = FPS * SEC  # 360 frames


def compute_features(df, joint_cols):
    """Build feature stack over the full joint_cols set.
    Missing columns are filled with NaN so SAITS can impute them."""
    n = len(df)
    features = np.full((n, len(joint_cols)), np.nan, dtype=np.float32)
    for i, col in enumerate(joint_cols):
        if col in df.columns:
            features[:, i] = df[col].values.astype(np.float32) / 100.0

    velocity = np.zeros_like(features)
    velocity[1:] = features[1:] - features[:-1]

    accel = np.zeros_like(features)
    accel[1:] = velocity[1:] - velocity[:-1]

    mean_pos = np.nanmean(features, axis=0, keepdims=True)
    residual = features - mean_pos

    return np.concatenate([features, residual, velocity, accel], axis=1)


def slice_trial(combined, window, step_sec):
    n = len(combined)
    step_size = step_sec * FPS
    return [combined[s:s + window] for s in range(0, n - window + 1, step_size)]


def run():
    print("Stage 07 s01: Slice body data per performer")
    print(f"  window={SEC}s ({WINDOW} frames), step={STEP}s")
    print(f"  performers: {len(PERFORMER_MAP)}")
    log_known_body_absences(printer=print)

    for (c, n), pieces in sorted(PERFORMER_MAP.items()):
        pid = performer_id(c, n)
        out_path = OUT_ROOT / pid / f"step{STEP}_sec{SEC}.npy"

        if out_path.exists():
            print(f"  {pid}: skip (exists)")
            continue

        print(f"\n  {pid}: {len(pieces)} pieces")
        all_slices = []
        for dataset, piece_prefix in tqdm(pieces, desc=f"    {pid}"):
            csv_path = get_piece_body_path(dataset, piece_prefix)
            if not csv_path.exists():
                continue
            df = pd.read_csv(csv_path)
            available = [col for col in body_joint_cols if col in df.columns]
            if len(available) < len(body_joint_cols) // 2:
                continue
            combined = compute_features(df, body_joint_cols)
            all_slices.extend(slice_trial(combined, WINDOW, STEP))

        if all_slices:
            arr = np.array(all_slices, dtype=np.float32)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            np.save(str(out_path), arr)
            print(f"    Saved: {arr.shape}")
        else:
            print(f"    No slices produced")


if __name__ == "__main__":
    run()
