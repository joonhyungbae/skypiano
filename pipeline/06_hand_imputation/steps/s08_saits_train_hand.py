"""Stage 06 Step 08: Train per-performer Left/Right hand SAITS.

s02 trains per-performer Body SAITS; s08 mirrors that for hands. Each
(performer, side) gets its own model — required because every pianist's
hand kinematics, marker drop pattern, and articulation differ enough that
a single global hand model washes out individual idiosyncrasies.

(Pre-2026-05-01 this trained ONE global Left and ONE global Right SAITS
across all performers; that design defect was corrected after curator
feedback that imputation must be per-performer for both body and hand.)

Input:  Data/Pipeline/06_hand_imputation/TrainData/{way}/{pid}/step1_sec3.npy   (s07 output)
Output: Data/Pipeline/06_hand_imputation/Models/{pid}/saits_{left,right}.pth
"""
from __future__ import annotations

import functools
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.performers import PERFORMER_MAP, performer_id

print = functools.partial(print, flush=True)

TRAIN_ROOT = Path("Data/Pipeline/06_hand_imputation/TrainData")
MODEL_ROOT = Path("Data/Pipeline/06_hand_imputation/Models")

SEC = 3
STEP = 1
EPOCHS = 1000
PATIENCE = 30


def run(only: list[str] | None = None, skip: list[str] | None = None,
        ways: list[str] | None = None,
        epochs: int = EPOCHS, patience: int = PATIENCE, force: bool = False,
        device: str | None = None):
    """Train per-performer Left/Right hand SAITS.

    Args:
        only: performer IDs to include (e.g. ["C0_N4", "C0_N5"]).
        skip: performer IDs to exclude.
        ways: subset of ["Left", "Right"]. Defaults to both.
        force: retrain even if a per-performer checkpoint already exists.
        device: 'cuda:0', 'cuda:1', or None (auto-pick first available CUDA).
                CPU is **never** used — the script aborts if no GPU is
                visible (caller must restrict via CUDA_VISIBLE_DEVICES if
                they only want one of multiple GPUs).
    """
    only_set = set(only) if only else None
    skip_set = set(skip) if skip else None
    ways = ways or ["Left", "Right"]

    print("Stage 06 s08: Train per-performer Left/Right hand SAITS")
    print(f"  ways={ways}  epochs={epochs}  patience={patience}")
    if only:
        print(f"  --only: {only}")
    if skip:
        print(f"  --skip: {skip}")

    try:
        from pypots.imputation import SAITS
        import torch
    except ImportError:
        raise SystemExit("ERROR: pypots not installed. pip install pypots")

    if not torch.cuda.is_available():
        raise SystemExit(
            "ERROR: CUDA not available. Hand SAITS training MUST run on a "
            "GPU; refusing to fall back to CPU. Check `nvidia-smi` and the "
            "torch build supports CUDA."
        )
    if device is None:
        device = "cuda:0"
    elif not device.startswith("cuda"):
        raise SystemExit(f"ERROR: device must be CUDA (got {device!r}); CPU training is forbidden.")
    print(f"  device: {device}")

    pids = []
    for (c, n), _ in sorted(PERFORMER_MAP.items()):
        pid = performer_id(c, n)
        if only_set and pid not in only_set:
            continue
        if skip_set and pid in skip_set:
            continue
        pids.append(pid)

    for pid in pids:
        for way in ways:
            npy_path = TRAIN_ROOT / way / pid / f"step{STEP}_sec{SEC}.npy"
            model_path = MODEL_ROOT / pid / f"saits_{way.lower()}.pth"

            if not npy_path.exists():
                print(f"  {pid}/{way}: skip (no training data at {npy_path})")
                continue
            if model_path.exists() and not force:
                print(f"  {pid}/{way}: skip (model exists at {model_path}; use --force to retrain)")
                continue

            data = np.load(str(npy_path))
            n_samples, n_steps, n_features = data.shape
            print(f"\n  {pid}/{way}: {data.shape}")

            saving_dir = MODEL_ROOT / pid / f"saits_{way.lower()}"
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
                saving_path=str(saving_dir),
                device=device,
            )

            saits.fit({"X": data})
            model_path.parent.mkdir(parents=True, exist_ok=True)
            saits.save(str(model_path), overwrite=True)
            print(f"    Saved: {model_path}")
            del saits


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*", default=None,
                    help="performer IDs to limit to (e.g. C0_N4 C0_N5)")
    ap.add_argument("--skip", nargs="*", default=None,
                    help="performer IDs to exclude")
    ap.add_argument("--ways", nargs="*", choices=["Left", "Right"], default=None,
                    help="restrict to a subset of hand sides (default: both)")
    ap.add_argument("--epochs", type=int, default=EPOCHS)
    ap.add_argument("--patience", type=int, default=PATIENCE)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--device", default=None,
                    help="cuda:0 / cuda:1 (default: cuda:0). CPU is forbidden.")
    args = ap.parse_args()
    run(only=args.only, skip=args.skip, ways=args.ways,
        epochs=args.epochs, patience=args.patience, force=args.force,
        device=args.device)
