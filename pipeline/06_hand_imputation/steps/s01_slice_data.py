"""
Stage 06 Step 01: Per-performer data slicing + feature computation

Collects every piece for each performer, splits them into 3-second windows, and
computes Residual, Velocity, Acceleration, Quaternion, and Distance features.

Input:  Data/Pipeline/04_piece_split/{dataset}/{piece_prefix}_hand.csv  (per-piece hand motion, 120fps)
Output: Data/Pipeline/06_hand_imputation/Sliced/{performer_id}/Body/step1_sec3.npy
"""
import functools
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.utils import body_joint_cols, body_connections
from src.utils import compute_three_joint_quaternions_body, compute_joint_distances_body
from src.performers import PERFORMER_MAP, performer_id, get_piece_hand_path

print = functools.partial(print, flush=True)

OUT_ROOT = Path("Data/Pipeline/06_hand_imputation/Sliced")

FPS = 120
SEC = 3
STEP = 1
WINDOW = FPS * SEC  # 360 frames


def compute_features(df, joint_cols, connections):
    n = len(df)
    features = df[joint_cols].values / 100.0

    velocity = np.zeros_like(features)
    velocity[1:] = features[1:] - features[:-1]

    accel = np.zeros_like(features)
    accel[1:] = velocity[1:] - velocity[:-1]

    mean_pos = np.nanmean(features, axis=0, keepdims=True)
    residual = features - mean_pos

    quat_list = []
    dist_list = []
    for i in range(n):
        row = df.iloc[i]
        q = compute_three_joint_quaternions_body(row, connections)
        d = compute_joint_distances_body(row, connections)
        quat_list.append(q)
        dist_list.append(d)

    quat_arr = np.nan_to_num(pd.DataFrame(quat_list).values.astype(np.float32), 0.0)
    dist_arr = np.nan_to_num(pd.DataFrame(dist_list).values.astype(np.float32), 0.0)

    combined = np.concatenate([
        features.astype(np.float32),
        residual.astype(np.float32),
        velocity.astype(np.float32),
        accel.astype(np.float32),
        quat_arr,
        dist_arr,
    ], axis=1)

    return combined


def slice_trial(combined, window, step_frames):
    n = len(combined)
    step_size = step_frames * FPS
    slices = []
    for start in range(0, n - window + 1, step_size):
        slices.append(combined[start:start + window])
    return slices


def run():
    print("Stage 06 s01: Slice data per performer (from Stage 04 per-piece output)")
    print(f"Window: {SEC}s ({WINDOW} frames), Step: {STEP}s")
    print(f"Performers: {len(PERFORMER_MAP)}")

    for (c, n), pieces in sorted(PERFORMER_MAP.items()):
        pid = performer_id(c, n)
        out_dir = OUT_ROOT / pid / "Body"
        out_path = out_dir / f"step{STEP}_sec{SEC}.npy"

        if out_path.exists():
            print(f"  {pid}: skip (exists)")
            continue

        print(f"\n  {pid}: {len(pieces)} pieces")

        all_slices = []
        for dataset, piece_prefix in tqdm(pieces, desc=f"    {pid}"):
            csv_path = get_piece_hand_path(dataset, piece_prefix)
            if not csv_path.exists():
                continue

            df = pd.read_csv(csv_path)
            available = [c for c in body_joint_cols if c in df.columns]
            if len(available) < len(body_joint_cols) // 2:
                continue

            combined = compute_features(df, available, body_connections)
            slices = slice_trial(combined, WINDOW, STEP)
            all_slices.extend(slices)

        if all_slices:
            arr = np.array(all_slices, dtype=np.float32)
            out_dir.mkdir(parents=True, exist_ok=True)
            np.save(str(out_path), arr)
            print(f"    Saved: {arr.shape}")
        else:
            print(f"    No slices produced")


if __name__ == "__main__":
    run()
