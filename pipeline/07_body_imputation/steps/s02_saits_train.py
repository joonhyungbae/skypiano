"""
Stage 07 Step 02: per-performer SAITS training for full-body imputation.

Input:  Data/Pipeline/07_body_imputation/Sliced/{pid}/step1_sec3.npy
Output: Data/Pipeline/07_body_imputation/Models/{pid}/saits_body.pypots
"""
import functools
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.performers import PERFORMER_MAP, performer_id

print = functools.partial(print, flush=True)

SLICE_ROOT = Path("Data/Pipeline/07_body_imputation/Sliced")
MODEL_ROOT = Path("Data/Pipeline/07_body_imputation/Models")

SEC = 3
STEP = 1
EPOCHS = 1000
PATIENCE = 30


def run(only=None, skip=None, force=False, epochs=EPOCHS, patience=PATIENCE):
    print("Stage 07 s02: Train SAITS per performer (full body)")
    print(f"  epochs={epochs}, patience={patience}")
    if only: print(f"  only: {only}")
    if skip: print(f"  skip: {skip}")
    if force: print("  force: re-training existing models")

    try:
        from pypots.imputation import SAITS
        import torch
    except ImportError:
        print("ERROR: pypots not installed. pip install pypots")
        return

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"  device: {device}")
    if device == "cpu":
        print("  WARN: CUDA not available — CPU training will be slow")

    for (c, n), pieces in sorted(PERFORMER_MAP.items()):
        pid = performer_id(c, n)
        if only and pid not in only: continue
        if skip and pid in skip:
            print(f"  {pid}: skip (excluded)")
            continue

        npy_path = SLICE_ROOT / pid / f"step{STEP}_sec{SEC}.npy"
        model_dir = MODEL_ROOT / pid

        if (model_dir / "saits_body.pypots").exists() and not force:
            print(f"  {pid}: skip (model exists, use --force)")
            continue
        if not npy_path.exists():
            print(f"  {pid}: skip (no sliced data)")
            continue

        data = np.load(str(npy_path))
        n_samples, n_steps, n_features = data.shape
        print(f"\n  {pid}: {data.shape} ({len(pieces)} pieces)")
        if n_samples < 2:
            print(f"    WARNING: only {n_samples} slices, may train poorly")

        model_dir.mkdir(parents=True, exist_ok=True)
        saits = SAITS(
            n_steps=n_steps,
            n_features=n_features,
            n_layers=2,
            d_model=256,
            n_heads=4,
            d_k=64,
            d_v=64,
            d_ffn=128,
            dropout=0.1,
            epochs=epochs,
            patience=patience,
            batch_size=min(32, n_samples),
            saving_path=str(model_dir / "saits_body"),
            device=device,
        )
        saits.fit({"X": data})
        saits.save(str(model_dir / "saits_body.pypots"), overwrite=True)
        print(f"    Saved: {model_dir / 'saits_body.pypots'}")
        del saits


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*")
    ap.add_argument("--skip", nargs="*")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--epochs", type=int, default=EPOCHS)
    ap.add_argument("--patience", type=int, default=PATIENCE)
    args = ap.parse_args()
    run(only=args.only, skip=args.skip, force=args.force,
        epochs=args.epochs, patience=args.patience)
