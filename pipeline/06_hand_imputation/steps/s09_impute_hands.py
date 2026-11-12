"""Stage 06 Step 09: per-performer Left/Right hand imputation.

Loads each performer's own SAITS Left/Right model (from s08) and applies
it to that performer's per-piece sliced inputs. Pre-2026-05-01 a single
global SAITS-Left + SAITS-Right was applied to all performers; the
per-performer correction reflects the rule that every imputation model
must be trained on a single performer's data so it captures that
pianist's idiosyncrasies (kinematics, articulation, marker pattern).
"""
from __future__ import annotations

import os
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.performers import PERFORMER_MAP, performer_id  # noqa: E402
from src.utils import *  # noqa: E402,F401,F403
from src.visualization import *  # noqa: E402,F401,F403

warnings.filterwarnings("ignore", category=FutureWarning)

FPS = 120
SEC = 3
STEP = 1


def _model_path_for(pid: str, way: str) -> str | None:
    """Return the per-performer SAITS checkpoint for (pid, way), or None
    if neither legacy .pth nor .pypots exists."""
    base = Path(f"Data/Pipeline/06_hand_imputation/Models/{pid}/saits_{way.lower()}.pth")
    if base.exists():
        return str(base)
    pypots = base.with_suffix(base.suffix + ".pypots")
    if pypots.exists():
        return str(pypots)
    return None


def _filepath_lst(folder_path: str, sec: int) -> list[str]:
    sub = os.path.join(folder_path, f"sec{sec}")
    return [f.path for f in os.scandir(sub) if f.is_dir()]


def _impute_one_piece(saits, way: str, pid: str, folder_path: str,
                      target_cols: list[str], imp_order: list[str]) -> None:
    filename = os.path.basename(folder_path.rstrip("/\\"))
    existing_out = f"Data/Pipeline/06_hand_imputation/Imp/result/{way}/{pid}/{filename}.csv"
    if os.path.exists(existing_out):
        return

    vanila_path = f"Data/Pipeline/06_hand_imputation/AD/Result/{way}/{pid}/{filename}.csv"
    vanila_data = pd.read_csv(vanila_path)
    left_frames = len(vanila_data) % (SEC * FPS)
    start_idx = (SEC * FPS) - left_frames

    motion_files = sorted(makeMotionFilePathLst(folder_path, ".npy"))  # noqa: F405

    save_dir = f"Data/Pipeline/06_hand_imputation/Imp/result/{way}/{pid}/"
    o_save = f"Data/Pipeline/06_hand_imputation/Imp/originalResult/{way}/{pid}/"
    n_save = f"Data/Pipeline/06_hand_imputation/Imp/naiveResult/{way}/{pid}/"
    for d in (save_dir, o_save, n_save):
        os.makedirs(d, exist_ok=True)

    naiveDF = pd.DataFrame(columns=target_cols)
    resultDF = pd.DataFrame(columns=target_cols)
    originalDF = pd.DataFrame(columns=target_cols)

    for num, path in enumerate(motion_files):
        data = np.load(path)
        original = pd.DataFrame(data).iloc[:, : len(target_cols)]
        original.columns = target_cols
        feature = pd.DataFrame(data).iloc[:, len(target_cols):]
        is_last = num == len(motion_files) - 1

        if is_last:
            originalDF = pd.concat([originalDF, original.iloc[start_idx:]])
        else:
            originalDF = pd.concat([originalDF, original])

        for idx, joint in enumerate(imp_order):
            arr = np.expand_dims(data, axis=0)
            imputed = saits.impute({"X": arr})
            imputed_arr = np.array([imputed])
            squeezed = (
                np.squeeze(imputed_arr, axis=1)
                  .reshape(-1, imputed_arr.shape[3])
            )
            df = pd.DataFrame(squeezed).iloc[:, : len(target_cols)]
            df.columns = target_cols

            if idx == 0:
                naiveDF = pd.concat(
                    [naiveDF, df.iloc[start_idx:] if is_last else df]
                )

            original[f"{joint}_x"] = df[f"{joint}_x"]
            original[f"{joint}_y"] = df[f"{joint}_y"]
            original[f"{joint}_z"] = df[f"{joint}_z"]
            data = np.array(pd.concat([original, feature], axis=1))

        if is_last:
            original = original.iloc[start_idx:]
        resultDF = pd.concat([resultDF, original])

    resultDF.reset_index(drop=True).to_csv(f"{save_dir}/{filename}.csv", index=False)
    originalDF.reset_index(drop=True).to_csv(f"{o_save}/{filename}.csv", index=False)
    naiveDF.reset_index(drop=True).to_csv(f"{n_save}/{filename}.csv", index=False)
    print(f"  {pid}/{way}/{filename}: in={len(vanila_data)} out={len(originalDF)}")


def run(only: list[str] | None = None, skip: list[str] | None = None,
        ways: list[str] | None = None, device: str | None = None) -> None:
    only_set = set(only) if only else None
    skip_set = set(skip) if skip else None
    ways = ways or ["Left", "Right"]

    try:
        from pypots.imputation import SAITS  # type: ignore
        import torch
    except ImportError:
        raise SystemExit("ERROR: pypots not installed. pip install pypots")

    if not torch.cuda.is_available():
        raise SystemExit(
            "ERROR: CUDA not available. Hand SAITS imputation MUST run on a "
            "GPU; refusing to fall back to CPU."
        )
    if device is None:
        device = "cuda:0"
    elif not device.startswith("cuda"):
        raise SystemExit(f"ERROR: device must be CUDA (got {device!r}); CPU inference is forbidden.")

    print("Stage 06 s09: per-performer hand imputation")
    print(f"  device: {device}")
    if only:
        print(f"  --only: {only}")
    if skip:
        print(f"  --skip: {skip}")

    for way in ways:
        if way == "Right":
            target_cols = right_hand_joint_cols  # noqa: F405
            imp_order = ["RLM", "RRM", "RWrist", "RIM", "RLP", "RMM", "RRP", "RMP",
                         "RIP", "RTF", "RTM", "RTD", "RLD", "RRD", "RMD", "RTC",
                         "RLF", "RID", "RRF", "RMF", "RIF"]
        else:
            target_cols = left_hand_joint_cols  # noqa: F405
            imp_order = ["LIM", "LWrist", "LLM", "LMM", "LRM", "LLP", "LTM", "LRP",
                         "LIP", "LMP", "LLD", "LTD", "LRD", "LID", "LTF", "LTC",
                         "LLF", "LMD", "LIF", "LRF", "LMF"]

        for (c_val, n_val), _ in sorted(PERFORMER_MAP.items()):
            pid = performer_id(c_val, n_val)
            if only_set and pid not in only_set:
                continue
            if skip_set and pid in skip_set:
                continue

            sliced_root = f"Data/Pipeline/06_hand_imputation/Imp/SlicedNonOverlapped/{way}/{pid}"
            if not os.path.isdir(os.path.join(sliced_root, f"sec{SEC}")):
                print(f"  {pid}/{way}: skip (no SlicedNonOverlapped data)")
                continue

            model_path = _model_path_for(pid, way)
            if not model_path:
                print(f"  {pid}/{way}: skip (no per-performer model — train s08 first)")
                continue

            # SAITS expects n_steps/n_features at init time; peek at one
            # sliced npy to derive them rather than relying on a defunct
            # global TrainData/{way} npy.
            example_dir = next(
                (d.path for d in os.scandir(os.path.join(sliced_root, f"sec{SEC}")) if d.is_dir()),
                None,
            )
            example_npy = next(
                (Path(example_dir).glob("*.npy")), None,
            ) if example_dir else None
            if not example_npy:
                print(f"  {pid}/{way}: skip (no sliced npy under {sliced_root})")
                continue
            shape = np.load(str(example_npy)).shape
            n_steps, n_features = shape[0], shape[1]

            saits = SAITS(
                n_steps=n_steps, n_features=n_features,
                n_layers=2, d_model=256, n_heads=4,
                d_k=64, d_v=64, d_ffn=128,
                dropout=0.1, batch_size=32,
                device=device,
            )
            saits.load(model_path)
            print(f"  {pid}/{way}: loaded {model_path}")

            for folder in tqdm(_filepath_lst(sliced_root, SEC),
                               desc=f"  {pid}/{way}"):
                _impute_one_piece(saits, way, pid, folder, target_cols, imp_order)
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
    ap.add_argument("--device", default=None,
                    help="cuda:0 / cuda:1 (default: cuda:0). CPU is forbidden.")
    args = ap.parse_args()
    run(only=args.only, skip=args.skip, ways=args.ways, device=args.device)
