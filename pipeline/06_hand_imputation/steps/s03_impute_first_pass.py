"""
Stage 06 Step 03: Per-performer first-pass SAITS imputation (Stage 04 per-piece based)

Fills in missing values for each performer's pieces using that performer's model.

Input:
  - Data/Pipeline/04_piece_split/{dataset}/{piece_prefix}_hand.csv
  - Data/Pipeline/06_hand_imputation/Models/{performer_id}/saits_body.pypots
Output:
  - Data/Pipeline/06_hand_imputation/Imp/result/Body/{performer_id}/{piece_prefix}.csv
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

MODEL_ROOT = Path("Data/Pipeline/06_hand_imputation/Models")
OUT_ROOT = Path("Data/Pipeline/06_hand_imputation/Imp/result/Body")

FPS = 120
SEC = 3
WINDOW = FPS * SEC

# Optional GPU device override propagated from CLI (`--device`). Module-global
# so the per-performer loop can read it without threading the parameter.
_DEVICE_OVERRIDE: str | None = None


def compute_features(df, joint_cols, connections):
    n = len(df)
    features = df[joint_cols].values / 100.0
    velocity = np.zeros_like(features); velocity[1:] = features[1:] - features[:-1]
    accel = np.zeros_like(features); accel[1:] = velocity[1:] - velocity[:-1]
    mean_pos = np.nanmean(features, axis=0, keepdims=True)
    residual = features - mean_pos

    quat_list, dist_list = [], []
    for i in range(n):
        row = df.iloc[i]
        quat_list.append(compute_three_joint_quaternions_body(row, connections))
        dist_list.append(compute_joint_distances_body(row, connections))

    quat_arr = np.nan_to_num(pd.DataFrame(quat_list).values.astype(np.float32), 0.0)
    dist_arr = np.nan_to_num(pd.DataFrame(dist_list).values.astype(np.float32), 0.0)

    return np.concatenate([
        features.astype(np.float32), residual.astype(np.float32),
        velocity.astype(np.float32), accel.astype(np.float32),
        quat_arr, dist_arr,
    ], axis=1)


def run(only: list[str] | None = None, skip: list[str] | None = None):
    print("Stage 06 s03: First-pass imputation per performer (Stage 04 per-piece)")
    if only:
        print(f"  only: {only}")
    if skip:
        print(f"  skip: {skip}")

    try:
        from pypots.imputation import SAITS
    except ImportError:
        print("ERROR: pypots not installed.")
        return

    for (c, n), pieces in sorted(PERFORMER_MAP.items()):
        pid = performer_id(c, n)
        if only and pid not in only:
            continue
        if skip and pid in skip:
            print(f"  {pid}: skip (excluded by flag)")
            continue
        model_path = MODEL_ROOT / pid / "saits_body.pypots"

        if not model_path.exists():
            print(f"  {pid}: skip (no model)")
            continue

        out_dir = OUT_ROOT / pid
        remaining = [(ds, pp) for ds, pp in pieces
                     if not (out_dir / f"{pp}.csv").exists()]
        if not remaining:
            print(f"  {pid}: skip (all done)")
            continue
        print(f"\n  {pid}: {len(remaining)}/{len(pieces)} to process")
        # pypots 1.3: instantiate first, then load weights
        # Hyperparameters must match s02_saits_train.py
        import torch
        if not torch.cuda.is_available():
            raise SystemExit(
                "ERROR: CUDA not available. Body imputation MUST run on GPU; "
                "refusing to fall back to CPU."
            )
        device = _DEVICE_OVERRIDE or "cuda:0"
        saits = SAITS(
            n_steps=WINDOW,
            n_features=152,  # matches slice data feature dim
            n_layers=2,
            d_model=256,
            n_heads=4,
            d_k=64,
            d_v=64,
            d_ffn=128,
            dropout=0.1,
            epochs=1,
            batch_size=32,
            device=device,
        )
        saits.load(str(model_path))

        out_dir.mkdir(parents=True, exist_ok=True)
        for dataset, piece_prefix in tqdm(remaining, desc=f"    {pid}"):
            csv_path = get_piece_hand_path(dataset, piece_prefix)
            if not csv_path.exists():
                continue

            df = pd.read_csv(csv_path)
            available = [col for col in body_joint_cols if col in df.columns]
            if len(available) < len(body_joint_cols) // 2:
                continue

            n_frames = len(df)
            combined = compute_features(df, available, body_connections)

            # Pad to multiple of WINDOW
            n_pad = (WINDOW - n_frames % WINDOW) % WINDOW
            if n_pad > 0:
                combined = np.concatenate([combined, np.full((n_pad, combined.shape[1]), np.nan)])

            n_windows = len(combined) // WINDOW
            windows = combined.reshape(n_windows, WINDOW, -1)

            result = saits.impute({"X": windows})
            imputed = result.reshape(-1, combined.shape[1])[:n_frames]

            imp_positions = imputed[:, :len(available)] * 100.0
            imp_df = pd.DataFrame(imp_positions, columns=available)
            imp_df.to_csv(str(out_dir / f"{piece_prefix}.csv"), index=False)

        del saits
        print(f"    Done: {out_dir}")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*", help="performer IDs to include")
    ap.add_argument("--skip", nargs="*", help="performer IDs to exclude")
    ap.add_argument("--device", default=None,
                    help="cuda:0 / cuda:1 (default: cuda:0). CPU is forbidden.")
    args = ap.parse_args()
    if args.device is not None and not args.device.startswith("cuda"):
        raise SystemExit(f"ERROR: device must be CUDA (got {args.device!r}); CPU is forbidden.")
    _DEVICE_OVERRIDE = args.device
    run(only=args.only, skip=args.skip)
