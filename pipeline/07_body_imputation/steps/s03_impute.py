"""
Stage 07 Step 03: per-performer SAITS imputation for full body.

Input:
  - Data/Pipeline/04_piece_split/{dataset}/{piece_prefix}_body.csv
  - Data/Pipeline/07_body_imputation/Models/{pid}/saits_body.pypots
  - Data/Pipeline/07_body_imputation/Sliced/{pid}/step1_sec3.npy  (for n_features)
Output:
  - Data/Pipeline/07_body_imputation/Imp/result/{pid}/{piece_prefix}.csv
"""
import functools
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.utils import body_joint_cols
from src.performers import PERFORMER_MAP, performer_id, get_piece_body_path

print = functools.partial(print, flush=True)

SLICE_ROOT = Path("Data/Pipeline/07_body_imputation/Sliced")
MODEL_ROOT = Path("Data/Pipeline/07_body_imputation/Models")
OUT_ROOT = Path("Data/Pipeline/07_body_imputation/Imp/result")

FPS = 120
SEC = 3

# Optional GPU device override propagated from CLI (`--device`).
_DEVICE_OVERRIDE: str | None = None
STEP = 1
WINDOW = FPS * SEC


def compute_features(df, joint_cols):
    """Build feature stack over the full joint_cols set.
    Missing columns are filled with NaN so SAITS can impute them."""
    n = len(df)
    features = np.full((n, len(joint_cols)), np.nan, dtype=np.float32)
    for i, col in enumerate(joint_cols):
        if col in df.columns:
            features[:, i] = df[col].values.astype(np.float32) / 100.0
    velocity = np.zeros_like(features); velocity[1:] = features[1:] - features[:-1]
    accel = np.zeros_like(features); accel[1:] = velocity[1:] - velocity[:-1]
    mean_pos = np.nanmean(features, axis=0, keepdims=True)
    residual = features - mean_pos
    return np.concatenate([features, residual, velocity, accel], axis=1)


def run(only=None, skip=None):
    print("Stage 07 s03: Per-performer body imputation")
    if only: print(f"  only: {only}")
    if skip: print(f"  skip: {skip}")

    try:
        from pypots.imputation import SAITS
        import torch
    except ImportError:
        raise SystemExit("ERROR: pypots not installed.")

    if not torch.cuda.is_available():
        raise SystemExit(
            "ERROR: CUDA not available. Body imputation MUST run on GPU; "
            "refusing to fall back to CPU."
        )
    device = _DEVICE_OVERRIDE or "cuda:0"

    for (c, n), pieces in sorted(PERFORMER_MAP.items()):
        pid = performer_id(c, n)
        if only and pid not in only: continue
        if skip and pid in skip:
            print(f"  {pid}: skip (excluded)")
            continue

        model_path = MODEL_ROOT / pid / "saits_body.pypots"
        slice_path = SLICE_ROOT / pid / f"step{STEP}_sec{SEC}.npy"
        if not model_path.exists():
            print(f"  {pid}: skip (no model)")
            continue
        if not slice_path.exists():
            print(f"  {pid}: skip (no sliced reference)")
            continue

        out_dir = OUT_ROOT / pid
        remaining = [(ds, pp) for ds, pp in pieces
                     if not (out_dir / f"{pp}.csv").exists()]
        if not remaining:
            print(f"  {pid}: skip (all done)")
            continue
        print(f"\n  {pid}: {len(remaining)}/{len(pieces)} to process")

        n_features = int(np.load(str(slice_path), mmap_mode="r").shape[2])
        saits = SAITS(
            n_steps=WINDOW,
            n_features=n_features,
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
            csv_path = get_piece_body_path(dataset, piece_prefix)
            if not csv_path.exists():
                continue
            df = pd.read_csv(csv_path)
            available = [col for col in body_joint_cols if col in df.columns]
            if len(available) < len(body_joint_cols) // 2:
                continue

            n_frames = len(df)
            combined = compute_features(df, body_joint_cols)

            n_pad = (WINDOW - n_frames % WINDOW) % WINDOW
            if n_pad > 0:
                combined = np.concatenate(
                    [combined, np.full((n_pad, combined.shape[1]), np.nan)]
                )

            n_windows = len(combined) // WINDOW
            windows = combined.reshape(n_windows, WINDOW, -1)
            result = saits.impute({"X": windows})
            imputed = result.reshape(-1, combined.shape[1])[:n_frames]

            imp_positions = imputed[:, :len(body_joint_cols)] * 100.0
            pd.DataFrame(imp_positions, columns=body_joint_cols).to_csv(
                str(out_dir / f"{piece_prefix}.csv"), index=False)

        del saits
        print(f"    Done: {out_dir}")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*")
    ap.add_argument("--skip", nargs="*")
    ap.add_argument("--device", default=None,
                    help="cuda:0 / cuda:1 (default: cuda:0). CPU is forbidden.")
    args = ap.parse_args()
    if args.device is not None and not args.device.startswith("cuda"):
        raise SystemExit(f"ERROR: device must be CUDA (got {args.device!r}); CPU is forbidden.")
    _DEVICE_OVERRIDE = args.device
    run(only=args.only, skip=args.skip)
