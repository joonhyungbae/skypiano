"""
Stage 06 Step 06: Rule-based outlier detection (replaces TimesNet AD)

Detection rules (applied sequentially, without double-counting):
  1. Bone length (rigid segments only): per-performer deviation from median ± k*MAD → distal marker outlier
     — Distances that change with finger flexion (e.g. Wrist→MCP) are not treated as bones
     — MAD-based threshold (k=5 ≈ 3σ equivalent) reflects natural per-performer/per-bone variability
  2. Velocity: inter-frame displacement > 30mm (≈3.6m/s @ 120fps) → outlier
  3. Acceleration: 2nd derivative > 500mm/frame² → outlier
  4. Position bound: |p| > 220mm in wrist-normalized space → outlier (fingertip max ≈170mm)

Each rule counts **only frames/markers not yet flagged by an earlier rule** (deduplicated).

Input:
  - Data/Pipeline/06_hand_imputation/AD/Split/{way}/{pid}/{filename}.csv (Stage 06 s05 hand-split output)
Output:
  - Data/Pipeline/06_hand_imputation/AD/Result/{way}/{pid}/{filename}.csv (outliers replaced with NaN)
  - Data/Pipeline/06_hand_imputation/Metadata/Outlier/{way}/{pid}/{filename}.json (detection report)
  - Data/Pipeline/06_hand_imputation/Metadata/Outlier/baselines/{pid}_{way}.json (per-performer baseline)
"""
from __future__ import annotations

import functools
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.performers import PERFORMER_MAP, performer_id

print = functools.partial(print, flush=True)

SPLIT_ROOT = Path("Data/Pipeline/06_hand_imputation/AD/Split")
OUT_ROOT = Path("Data/Pipeline/06_hand_imputation/AD/Result")
META_ROOT = Path("Data/Pipeline/06_hand_imputation/Metadata/Outlier")

FPS = 120.0
WAYS = ["Left", "Right"]

# Rigid bone segments (distance stays nearly constant during playing).
# Wrist→MCP is NOT rigid (flex/extend changes distance), so it's excluded.
# Each finger: MCP→PIP→DIP→Tip. Thumb: CMC→MCP→IP→Tip (4 markers).
FINGER_BONES_REL = {
    "I": [("IM", "IP"), ("IP", "ID"), ("ID", "IF")],
    "L": [("LM", "LP"), ("LP", "LD"), ("LD", "LF")],
    "M": [("MM", "MP"), ("MP", "MD"), ("MD", "MF")],
    "R": [("RM", "RP"), ("RP", "RD"), ("RD", "RF")],
    "T": [("TC", "TM"), ("TM", "TD"), ("TD", "TF")],
}

# All distal finger markers we track (20 per hand)
DISTAL_MARKERS_REL = [b[1] for bones in FINGER_BONES_REL.values() for b in bones]


def bones_for(way: str) -> list[tuple[str, str]]:
    """Return prefixed bone pairs for a hand."""
    prefix = "R" if way == "Right" else "L"
    return [(f"{prefix}{a}", f"{prefix}{b}") for bones in FINGER_BONES_REL.values() for a, b in bones]


def distal_markers_for(way: str) -> list[str]:
    prefix = "R" if way == "Right" else "L"
    return [f"{prefix}{m}" for m in DISTAL_MARKERS_REL]


def marker_xyz(df: pd.DataFrame, marker: str) -> np.ndarray | None:
    cols = [f"{marker}_x", f"{marker}_y", f"{marker}_z"]
    if not all(c in df.columns for c in cols):
        return None
    return df[cols].values


def compute_bone_lengths(df: pd.DataFrame, bones: list[tuple[str, str]]) -> dict[tuple, np.ndarray]:
    """Compute per-frame bone lengths for each bone. Returns dict[bone → (n,)]."""
    result = {}
    for prox, dist in bones:
        p = marker_xyz(df, prox)
        d = marker_xyz(df, dist)
        if p is None or d is None:
            continue
        result[(prox, dist)] = np.linalg.norm(d - p, axis=1)
    return result


def compute_performer_baseline(pid: str, way: str) -> dict[tuple, tuple[float, float]] | None:
    """
    Compute per-bone robust baseline: (median, MAD).
    Uses all performer's pieces; filters NaN and zero lengths.
    MAD scaled by 1.4826 to approximate Gaussian std.
    """
    pid_dir = SPLIT_ROOT / way / pid
    if not pid_dir.is_dir():
        return None
    files = sorted(pid_dir.glob("*.csv"))
    if not files:
        return None

    bones = bones_for(way)
    all_lengths = {b: [] for b in bones}

    for f in files:
        df = pd.read_csv(f)
        lengths = compute_bone_lengths(df, bones)
        for b, arr in lengths.items():
            valid = arr[~np.isnan(arr) & (arr > 1e-3)]
            if len(valid):
                all_lengths[b].append(valid)

    baseline = {}
    for b, parts in all_lengths.items():
        if not parts:
            continue
        concat = np.concatenate(parts)
        med = float(np.median(concat))
        mad = float(np.median(np.abs(concat - med))) * 1.4826  # scaled to std
        # floor MAD so we don't get zero-tolerance from extreme imputation collapse
        mad = max(mad, med * 0.01)  # at least 1% of median
        baseline[b] = (med, mad)
    return baseline


def detect_outliers(
    df: pd.DataFrame,
    way: str,
    baseline: dict[tuple, tuple[float, float]],
    mad_k: float = 5.0,
    vel_thresh_mm: float = 30.0,
    accel_thresh_mm: float = 500.0,
    pos_max_mm: float = 220.0,
) -> tuple[pd.DataFrame, dict]:
    """
    Detect outliers with non-overlapping rule attribution.
    Returns (cleaned df with NaN at outlier positions, report).
    """
    n = len(df)
    bones = bones_for(way)
    hand_markers = distal_markers_for(way)

    cleaned = df.copy()

    # Per-marker cumulative outlier mask; a frame/marker becomes outlier if
    # ANY rule flags it. Rule hit counts attribute ONLY newly-flagged frames.
    cumulative_mask = {m: np.zeros(n, dtype=bool) for m in hand_markers}

    report = {
        "n_frames": n,
        "rule_hits": {"bone": 0, "velocity": 0, "accel": 0, "position": 0},
        "marker_outlier_count": {},
        "bone_stats": {},  # bone → max |z-score| observed (diagnostic)
    }

    def apply_rule(rule_name: str, per_marker_mask: dict[str, np.ndarray]):
        """OR into cumulative mask; count only NEW (previously unflagged) frames."""
        new_count = 0
        for m, mask in per_marker_mask.items():
            if m not in cumulative_mask:
                continue
            new_flags = mask & ~cumulative_mask[m]
            new_count += int(new_flags.sum())
            cumulative_mask[m] |= mask
        report["rule_hits"][rule_name] = new_count

    # ── Rule 1: bone length (MAD-based)
    bone_masks = {m: np.zeros(n, dtype=bool) for m in hand_markers}
    current_lengths = compute_bone_lengths(df, bones)
    for (prox, dist), arr in current_lengths.items():
        if (prox, dist) not in baseline:
            continue
        med, mad = baseline[(prox, dist)]
        if mad <= 0:
            continue
        z = np.abs(arr - med) / mad
        bad = (z > mad_k) & ~np.isnan(arr)
        if dist in bone_masks:
            bone_masks[dist] |= bad
        # diagnostic
        valid_z = z[~np.isnan(z) & np.isfinite(z)]
        report["bone_stats"][f"{prox}->{dist}"] = {
            "median_mm": round(med, 2),
            "mad_mm": round(mad, 2),
            "max_z": round(float(valid_z.max()), 2) if len(valid_z) else 0.0,
            "flagged": int(bad.sum()),
        }
    apply_rule("bone", bone_masks)

    # ── Rule 2: velocity
    vel_masks = {}
    for m in hand_markers:
        p = marker_xyz(df, m)
        if p is None:
            vel_masks[m] = np.zeros(n, dtype=bool)
            continue
        vel = np.zeros(n)
        vel[1:] = np.linalg.norm(np.diff(p, axis=0), axis=1)
        vel_masks[m] = (vel > vel_thresh_mm) & ~np.isnan(vel)
    apply_rule("velocity", vel_masks)

    # ── Rule 3: acceleration
    accel_masks = {}
    for m in hand_markers:
        p = marker_xyz(df, m)
        if p is None:
            accel_masks[m] = np.zeros(n, dtype=bool)
            continue
        accel = np.zeros(n)
        if n >= 3:
            accel[1:-1] = np.linalg.norm(p[2:] - 2 * p[1:-1] + p[:-2], axis=1)
        accel_masks[m] = (accel > accel_thresh_mm) & ~np.isnan(accel)
    apply_rule("accel", accel_masks)

    # ── Rule 4: position magnitude bound
    pos_masks = {}
    for m in hand_markers:
        p = marker_xyz(df, m)
        if p is None:
            pos_masks[m] = np.zeros(n, dtype=bool)
            continue
        mag = np.linalg.norm(p, axis=1)
        pos_masks[m] = (mag > pos_max_mm) & ~np.isnan(mag)
    apply_rule("position", pos_masks)

    # ── Apply cumulative outliers → NaN
    total = 0
    for m, mask in cumulative_mask.items():
        cnt = int(mask.sum())
        report["marker_outlier_count"][m] = cnt
        total += cnt
        if cnt > 0:
            for axis in "xyz":
                col = f"{m}_{axis}"
                if col in cleaned.columns:
                    cleaned.loc[mask, col] = np.nan
    report["total_outlier_frames"] = total
    report["outlier_rate_pct"] = round(100 * total / (n * len(hand_markers)), 3) if n > 0 else 0.0
    return cleaned, report


def run(only: list[str] | None = None, skip: list[str] | None = None):
    print("Stage 06 s06: Rule-based outlier detection (MAD-based, rigid bones only)")
    if only:
        print(f"  only: {only}")
    if skip:
        print(f"  skip: {skip}")

    baselines_dir = META_ROOT / "baselines"
    baselines_dir.mkdir(parents=True, exist_ok=True)

    for (c, n), _pieces in sorted(PERFORMER_MAP.items()):
        pid = performer_id(c, n)
        if only and pid not in only:
            continue
        if skip and pid in skip:
            print(f"  {pid}: skip (excluded by flag)")
            continue

        for way in WAYS:
            pid_dir = SPLIT_ROOT / way / pid
            if not pid_dir.is_dir():
                print(f"  {pid}/{way}: no split dir")
                continue

            files = sorted(pid_dir.glob("*.csv"))
            if not files:
                continue

            print(f"\n  {pid}/{way}: computing MAD baseline from {len(files)} pieces...")
            baseline = compute_performer_baseline(pid, way)
            if not baseline:
                print(f"    skipped: empty baseline")
                continue

            # Save baseline summary
            baseline_path = baselines_dir / f"{pid}_{way}.json"
            baseline_save = {
                f"{p}->{d}": {"median_mm": round(med, 3), "mad_mm": round(mad, 3)}
                for (p, d), (med, mad) in baseline.items()
            }
            with open(baseline_path, "w") as f:
                json.dump(baseline_save, f, indent=2)

            out_dir = OUT_ROOT / way / pid
            out_dir.mkdir(parents=True, exist_ok=True)
            meta_dir = META_ROOT / way / pid
            meta_dir.mkdir(parents=True, exist_ok=True)

            agg = {"n_files": 0, "rule_hits": {"bone": 0, "velocity": 0, "accel": 0, "position": 0},
                   "total_outliers": 0, "total_frames": 0}

            for f in tqdm(files, desc=f"    {pid}/{way}"):
                df = pd.read_csv(f)
                cleaned, report = detect_outliers(df, way, baseline)

                out_path = out_dir / f.name
                cleaned.to_csv(out_path, index=False)

                meta_path = meta_dir / f.name.replace(".csv", ".json")
                with open(meta_path, "w") as rf:
                    json.dump(report, rf, indent=2)

                agg["n_files"] += 1
                agg["total_frames"] += report["n_frames"]
                agg["total_outliers"] += report["total_outlier_frames"]
                for k, v in report["rule_hits"].items():
                    agg["rule_hits"][k] += v

            n_markers = len(distal_markers_for(way))
            rate = 100 * agg["total_outliers"] / max(1, agg["total_frames"] * n_markers)
            print(f"    Summary: {agg['n_files']} files, {agg['total_outliers']} outliers ({rate:.2f}%)")
            print(f"    Rules (non-overlapping): {agg['rule_hits']}")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*", help="performer IDs to include")
    ap.add_argument("--skip", nargs="*", help="performer IDs to exclude")
    args = ap.parse_args()
    run(only=args.only, skip=args.skip)
