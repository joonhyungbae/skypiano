"""
Stage 06 Step 07: Slice Left/Right hands (prepares input for hand-imputation training)

s06 produces AD/Result/{way}/{pid}/*.csv (hand data with outliers set to NaN).
The Stage 06 hand-imputation training input needs two kinds of data:

  1. Training data (overlapping, step=1s) — PER PERFORMER:
     Data/Pipeline/06_hand_imputation/TrainData/{way}/{pid}/step1_sec3.npy
     - One NPY per performer × hand side, capturing only that performer's
       motion idiosyncrasies (changed from a single global NPY on
       2026-05-01 to align with per-performer hand SAITS training).
     - 3-second (360 frame) window, 1-second (120 frame) step
     - shape: (n_windows_for_performer, 360, n_features)

  2. Inference data (non-overlapping, step=3s):
     Data/Pipeline/06_hand_imputation/Imp/SlicedNonOverlapped/{way}/{pid}/sec{sec}/{filename}/*.npy
     - Non-overlapping windows split per performer/piece (the hand-imputation step uses these for per-piece reconstruction)
     - One piece is split into multiple windows (in filename numeric order)

Input:
  - Data/Pipeline/06_hand_imputation/AD/Result/{way}/{pid}/*.csv
Output:
  - Data/Pipeline/06_hand_imputation/TrainData/{way}/{pid}/step1_sec3.npy   (per performer)
  - Data/Pipeline/06_hand_imputation/Imp/SlicedNonOverlapped/{way}/{pid}/sec3/{filename}/{idx}.npy
"""
from __future__ import annotations

import functools
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.performers import PERFORMER_MAP, performer_id

print = functools.partial(print, flush=True)

AD_ROOT = Path("Data/Pipeline/06_hand_imputation/AD/Result")
TRAIN_ROOT = Path("Data/Pipeline/06_hand_imputation/TrainData")
INFER_ROOT = Path("Data/Pipeline/06_hand_imputation/Imp/SlicedNonOverlapped")

FPS = 120
SEC = 3
STEP = 1
WINDOW = FPS * SEC  # 360


def hand_joint_cols(way: str) -> list[str]:
    """Return the 21-marker hand joint column list (x,y,z each = 63 cols)."""
    prefix = "R" if way == "Right" else "L"
    # 21 markers: Wrist + 5 fingers × 4 joints (MCP, PIP, DIP, Tip) = 21
    # Index/Little/Middle/Ring each have M/P/D/F; Thumb has TC/TM/TD/TF
    finger_joints = {
        "I": ["IM", "IP", "ID", "IF"],
        "L": ["LM", "LP", "LD", "LF"],
        "M": ["MM", "MP", "MD", "MF"],
        "R": ["RM", "RP", "RD", "RF"],
        "T": ["TC", "TM", "TD", "TF"],
    }
    markers = ["Wrist"]
    for f in ["I", "L", "M", "R", "T"]:
        markers.extend(finger_joints[f])
    cols = []
    for m in markers:
        for axis in "xyz":
            cols.append(f"{prefix}{m}_{axis}")
    return cols


def compute_features(positions: np.ndarray) -> np.ndarray:
    """
    Simple feature set: positions + velocity + acceleration.
    positions: (n_frames, n_joints*3)
    Returns: (n_frames, n_joints*3 * 3) — positions + velocity + accel concatenated
    """
    n = len(positions)
    vel = np.zeros_like(positions)
    vel[1:] = positions[1:] - positions[:-1]
    accel = np.zeros_like(positions)
    accel[1:] = vel[1:] - vel[:-1]
    return np.concatenate([positions, vel, accel], axis=1)


def slice_overlapping(features: np.ndarray, window: int, step: int) -> list[np.ndarray]:
    """Sliding window with given step. Returns list of (window, n_features) arrays."""
    n = len(features)
    slices = []
    for start in range(0, n - window + 1, step):
        slices.append(features[start:start + window])
    return slices


def slice_non_overlapping(features: np.ndarray, window: int) -> list[np.ndarray]:
    """
    Non-overlapping windows, with the LAST window right-aligned
    (backfills from end if there's a leftover tail — matches the hand-imputation expectation).

    n < window: front-pad with zeros to one full window. The hand-imputation step uses
    start_idx = window - (n % window) = window - n to crop the real tail.
    """
    n = len(features)
    if n < window:
        pad = np.zeros((window - n, features.shape[1]), dtype=features.dtype)
        return [np.concatenate([pad, features], axis=0)]

    slices = []
    n_full = n // window
    for i in range(n_full):
        slices.append(features[i * window:(i + 1) * window])
    if n % window != 0:
        slices.append(features[-window:])
    return slices


def build_train_data(way: str, only: list[str] | None = None, skip: list[str] | None = None):
    """For each performer: aggregate that performer's hand CSVs → sliced
    NPY for THAT performer only. Per-performer training file aligns with
    s08's per-performer SAITS training (one model per performer × side).

    Each piece-CSV is sliced independently with overlap; the resulting
    windows for the SAME performer (across all that performer's pieces)
    are stacked into one NPY. Piece boundaries don't carry through —
    SAITS training is unsupervised reconstruction, so collapsing all of
    a performer's available motion frames into a single training tensor
    is the right thing to do.
    """
    only_set = set(only) if only else None
    skip_set = set(skip) if skip else None
    cols = hand_joint_cols(way)
    base_dir = AD_ROOT / way
    if not base_dir.is_dir():
        return

    for (c, n), _pieces in sorted(PERFORMER_MAP.items()):
        pid = performer_id(c, n)
        if only_set and pid not in only_set:
            continue
        if skip_set and pid in skip_set:
            continue
        pid_dir = base_dir / pid
        if not pid_dir.is_dir():
            continue
        files = sorted(pid_dir.glob("*.csv"))
        if not files:
            continue

        per_performer_slices: list[np.ndarray] = []
        for f in tqdm(files, desc=f"  {way}/{pid} [train]"):
            df = pd.read_csv(f)
            present = [c_ for c_ in cols if c_ in df.columns]
            if len(present) < len(cols) * 0.8:
                continue
            for missing in [c_ for c_ in cols if c_ not in df.columns]:
                df[missing] = 0.0
            positions = df[cols].values.astype(np.float32)
            feats = compute_features(positions)
            per_performer_slices.extend(slice_overlapping(feats, WINDOW, STEP * FPS))

        if not per_performer_slices:
            print(f"  [train] {way}/{pid}: no usable windows — skipped")
            continue

        arr = np.stack(per_performer_slices, axis=0)
        out_dir = TRAIN_ROOT / way / pid
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"step{STEP}_sec{SEC}.npy"
        np.save(str(out_path), arr)
        print(f"  [train] {way}/{pid}: saved {arr.shape} → {out_path}")


def build_inference_data(way: str, only: list[str] | None = None, skip: list[str] | None = None):
    """For each performer's piece CSV: non-overlapping windows to per-file subfolder."""
    only_set = set(only) if only else None
    skip_set = set(skip) if skip else None
    cols = hand_joint_cols(way)

    for (c, n), _pieces in sorted(PERFORMER_MAP.items()):
        pid = performer_id(c, n)
        if only_set and pid not in only_set:
            continue
        if skip_set and pid in skip_set:
            continue
        pid_dir = AD_ROOT / way / pid
        if not pid_dir.is_dir():
            continue
        files = sorted(pid_dir.glob("*.csv"))

        for f in tqdm(files, desc=f"  {way}/{pid} [infer]"):
            df = pd.read_csv(f)
            for missing in [c_ for c_ in cols if c_ not in df.columns]:
                df[missing] = 0.0
            positions = df[cols].values.astype(np.float32)
            feats = compute_features(positions)

            slices = slice_non_overlapping(feats, WINDOW)
            if not slices:
                continue

            stem = f.stem
            out_dir = INFER_ROOT / way / pid / f"sec{SEC}" / stem
            out_dir.mkdir(parents=True, exist_ok=True)
            for i, sl in enumerate(slices):
                np.save(str(out_dir / f"{i:04d}.npy"), sl)


def run(only: list[str] | None = None, skip: list[str] | None = None,
        train_only: bool = False, infer_only: bool = False):
    print("Stage 06 s07: Slice L/R hands for hand-imputation training")
    print(f"  Window: {SEC}s ({WINDOW} frames), Train step: {STEP}s, Infer step: non-overlapping")
    if only:
        print(f"  --only: {only}")
    if skip:
        print(f"  --skip: {skip}")

    for way in ["Left", "Right"]:
        if not infer_only:
            print(f"\n  [{way}] Building per-performer training data...")
            build_train_data(way, only=only, skip=skip)
        if not train_only:
            print(f"\n  [{way}] Building inference data (per-performer, non-overlapping)...")
            build_inference_data(way, only=only, skip=skip)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*", default=None,
                    help="performer IDs to limit to (e.g. C0_N4 C0_N5)")
    ap.add_argument("--skip", nargs="*", default=None,
                    help="performer IDs to exclude")
    ap.add_argument("--train-only", action="store_true")
    ap.add_argument("--infer-only", action="store_true")
    args = ap.parse_args()
    run(only=args.only, skip=args.skip,
        train_only=args.train_only, infer_only=args.infer_only)
