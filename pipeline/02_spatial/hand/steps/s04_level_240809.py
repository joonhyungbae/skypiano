"""
Step 4: Level Reference Data (240809) — Horizontal Alignment

Makes the piano keyboard plane perfectly horizontal (Z = constant) for 240809.
Uses 4 end markers (LEnd1, LEnd, REnd1, REnd) for SVD plane fitting.

Saves reference parameters for Step 5.

Input:  Data/Pipeline/02_spatial/hand/Centered/240809/*.csv
Output: Data/Pipeline/02_spatial/hand/Aligned/240809/*.csv
        Data/Pipeline/02_spatial/hand/Aligned/reference_params.npz
"""

import pandas as pd
import numpy as np
import os
from tqdm import tqdm

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.transform import (
    compute_plane_leveling_rotation,
    compute_per_frame_rotation,
    smooth_rotations,
)
from src.io_utils import (
    get_stage_files, get_output_dir, load_csv, save_csv,
    get_four_end_markers, apply_transform_to_df, apply_per_frame_transform_to_df,
)
from src.markers import get_marker_names

Z_STD_THRESHOLD = 0.5
SMOOTHING_WINDOW = 5


def calculate_file_z_std(df):
    z_stds = []
    for idx in range(len(df)):
        points, valid = get_four_end_markers(df, idx)
        if valid:
            z_stds.append(np.std(points[:, 2]))
    return np.mean(z_stds) if z_stds else float('inf')


def level_file_uniform(df):
    mid = len(df) // 2
    for offset in range(len(df) // 2):
        for idx in [mid + offset, mid - offset]:
            if 0 <= idx < len(df):
                points, valid = get_four_end_markers(df, idx)
                if valid and len(points) >= 3:
                    R_mat = compute_plane_leveling_rotation(points)
                    df_leveled = apply_transform_to_df(df, R_mat)
                    return df_leveled, calculate_file_z_std(df_leveled)
    return df, float('inf')


from steps.s05_cross_year_align import ensure_canonical_orientation  # reused post-align fix


def level_file_per_frame(df):
    rotations = []
    for idx in range(len(df)):
        points, valid = get_four_end_markers(df, idx)
        if valid and len(points) >= 3:
            rotations.append(compute_per_frame_rotation(points))
        else:
            rotations.append(np.eye(3))
    rotations = smooth_rotations(rotations, window=SMOOTHING_WINDOW)
    df_leveled = apply_per_frame_transform_to_df(df, rotations)
    return df_leveled, calculate_file_z_std(df_leveled)


def save_reference_params(aligned_files, output_base):
    all_lend, all_rend = [], []
    for path in aligned_files:
        df = load_csv(path)
        lend = df[['LEnd_x', 'LEnd_y', 'LEnd_z']].median().values
        rend = df[['REnd_x', 'REnd_y', 'REnd_z']].median().values
        if not np.isnan(lend).any() and not np.isnan(rend).any():
            all_lend.append(lend)
            all_rend.append(rend)

    ref_lend = np.median(all_lend, axis=0)
    ref_rend = np.median(all_rend, axis=0)
    ref_vector = ref_rend - ref_lend
    ref_vector_unit = ref_vector / np.linalg.norm(ref_vector)

    params = {
        'ref_lend': ref_lend, 'ref_rend': ref_rend,
        'ref_vector': ref_vector, 'ref_vector_unit': ref_vector_unit,
        'ref_midpoint': (ref_lend + ref_rend) / 2,
    }
    ref_path = os.path.join(output_base, 'reference_params.npz')
    np.savez(ref_path, **params)
    print(f"\n  Reference saved: LEnd={ref_lend.round(2)}, REnd={ref_rend.round(2)}")


def run():
    print("\n" + "=" * 60)
    print("  STEP 4: Level 240809 (Horizontal Alignment)")
    print("=" * 60)

    input_files = get_stage_files('Centered', '240809')
    output_dir = get_output_dir('Aligned', '240809')

    if not input_files:
        print("  No 240809 files found")
        return

    results = []
    needs_per_frame = []

    print(f"\n  Pass 1: File-level rotation ({len(input_files)} files)")
    for path in tqdm(input_files, desc="  File-level"):
        filename = os.path.basename(path)
        df = load_csv(path)
        z_before = calculate_file_z_std(df)
        df_leveled, z_after = level_file_uniform(df)

        if z_after > Z_STD_THRESHOLD:
            needs_per_frame.append((path, filename))
        save_csv(df_leveled, os.path.join(output_dir, filename))
        results.append((filename, z_before, z_after, 'OK' if z_after <= Z_STD_THRESHOLD else 'NEED_PF'))

    if needs_per_frame:
        print(f"\n  Pass 2: Per-frame rotation ({len(needs_per_frame)} files)")
        for path, filename in tqdm(needs_per_frame, desc="  Per-frame"):
            df = load_csv(os.path.join(output_dir, filename))
            df_leveled, z_after = level_file_per_frame(df)
            save_csv(df_leveled, os.path.join(output_dir, filename))
            for i, (fn, zb, za, _) in enumerate(results):
                if fn == filename:
                    results[i] = (fn, zb, z_after, 'PER_FRAME')

    # Pass 3: Canonical orientation (wrists in +depth/+height quadrant)
    print(f"\n  Pass 3: Canonical orientation snap")
    snap_count = 0
    for path in get_stage_files('Aligned', '240809'):
        filename = os.path.basename(path)
        df = load_csv(path)
        df, snap_deg = ensure_canonical_orientation(df)
        if snap_deg != 0:
            save_csv(df, path)
            snap_count += 1
            print(f"    SNAP {snap_deg:+.0f}°: {filename}")
    print(f"    {snap_count} files snapped")

    print(f"\n  {'File':<30} {'Before':>8} {'After':>8} {'Method':>12}")
    print(f"  {'-'*60}")
    for fn, zb, za, m in sorted(results, key=lambda x: -x[2]):
        s = 'O' if za <= Z_STD_THRESHOLD else '!'
        print(f"  {s} {fn:<28} {zb:>8.3f} {za:>8.3f} {m:>12}")

    save_reference_params(get_stage_files('Aligned', '240809'), get_output_dir('Aligned'))
    print("\n  Step 4 complete.")


if __name__ == '__main__':
    run()
