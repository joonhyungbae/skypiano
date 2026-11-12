"""
Stage 06 Step 02: Per-performer SAITS model training (Stage 04 per-piece based)

Trains a separate SAITS model on each performer's sliced data.
(Bone lengths differ across performers, so per-performer models are required.)

Input:  Data/Pipeline/06_hand_imputation/Sliced/{performer_id}/Body/step1_sec3.npy
Output: Data/Pipeline/06_hand_imputation/Models/{performer_id}/saits_body/
"""
import functools
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.performers import PERFORMER_MAP, performer_id

print = functools.partial(print, flush=True)

SLICE_ROOT = Path("Data/Pipeline/06_hand_imputation/Sliced")
MODEL_ROOT = Path("Data/Pipeline/06_hand_imputation/Models")

SEC = 3
STEP = 1
EPOCHS = 1000
PATIENCE = 30


def run(only: list[str] | None = None, skip: list[str] | None = None,
        force: bool = False, epochs: int = EPOCHS, patience: int = PATIENCE,
        device: str | None = None):
    """
    only: list of performer IDs to include (e.g. ['C0_N4', 'C0_N5'])
    skip: list of performer IDs to exclude
    force: re-train even if model exists
    epochs: max training epochs (default 1000)
    patience: early stopping patience (default 30)
    device: 'cuda:0' / 'cuda:1'. CPU is forbidden.
    """
    print("Stage 06 s02: Train SAITS model per performer")
    print(f"  epochs={epochs}, patience={patience}")
    if only:
        print(f"  only: {only}")
    if skip:
        print(f"  skip: {skip}")
    if force:
        print(f"  force: re-training existing models")

    try:
        from pypots.imputation import SAITS
        import torch
    except ImportError:
        raise SystemExit("ERROR: pypots not installed. pip install pypots")

    if not torch.cuda.is_available():
        raise SystemExit(
            "ERROR: CUDA not available. Body SAITS training MUST run on a "
            "GPU; refusing to fall back to CPU."
        )
    if device is None:
        device = "cuda:0"
    elif not device.startswith("cuda"):
        raise SystemExit(f"ERROR: device must be CUDA (got {device!r}); CPU training is forbidden.")
    print(f"  device: {device}")

    for (c, n), pieces in sorted(PERFORMER_MAP.items()):
        pid = performer_id(c, n)
        if only and pid not in only:
            continue
        if skip and pid in skip:
            print(f"  {pid}: skip (excluded by flag)")
            continue
        npy_path = SLICE_ROOT / pid / "Body" / f"step{STEP}_sec{SEC}.npy"
        model_dir = MODEL_ROOT / pid

        if (model_dir / "saits_body.pypots").exists() and not force:
            print(f"  {pid}: skip (model exists, use --force to re-train)")
            continue

        if not npy_path.exists():
            print(f"  {pid}: skip (no sliced data)")
            continue

        data = np.load(str(npy_path))
        n_samples, n_steps, n_features = data.shape
        print(f"\n  {pid}: {data.shape} ({len(pieces)} pieces)")

        if n_samples < 2:
            print(f"    WARNING: only {n_samples} slices, may train poorly")

        # SAITS training
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

        dataset = {"X": data}
        saits.fit(dataset)
        saits.save(str(model_dir / "saits_body.pypots"), overwrite=True)
        print(f"    Model saved: {model_dir / 'saits_body.pypots'}")

        del saits


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*", help="performer IDs to include")
    ap.add_argument("--skip", nargs="*", help="performer IDs to exclude")
    ap.add_argument("--force", action="store_true", help="re-train even if model exists")
    ap.add_argument("--epochs", type=int, default=EPOCHS, help=f"max epochs (default {EPOCHS})")
    ap.add_argument("--patience", type=int, default=PATIENCE, help=f"early stopping patience (default {PATIENCE})")
    ap.add_argument("--device", default=None,
                    help="cuda:0 / cuda:1 (default: cuda:0). CPU is forbidden.")
    args = ap.parse_args()
    run(only=args.only, skip=args.skip, force=args.force,
        epochs=args.epochs, patience=args.patience, device=args.device)
