"""
Step 5: Cross-Year Alignment (all datasets → 240809)

Strategy:
  4-point datasets (240828, 240523):
    1. 2-point align to 240809 (direction + translation)
    2. Self-level using own 4 end markers (fixes roll)
    3. Z-flip check (ensure hands above keyboard)

  2-point datasets (211215, 220819, 221112):
    1. 2-point align to 240809 (direction + translation, NO roll heuristic)
    2. Analytical roll correction using elbow-wrist "up" vector
    3. Z-flip check

Analytical roll correction:
  After 2-point alignment, keyboard is along X.
  The remaining DOF is roll around X (Y-Z plane rotation).
  We use the anatomical prior: Elbow is ABOVE Wrist.
  The elbow-wrist vector, projected perpendicular to keyboard axis,
  should point in +Z direction. We compute the angle and rotate.

Input:  Data/Pipeline/02_spatial/hand/Centered/{datasets}/*.csv
        Data/Pipeline/02_spatial/hand/Aligned/reference_params.npz
Output: Data/Pipeline/02_spatial/hand/Aligned/{datasets}/*.csv
"""

import pandas as pd
import numpy as np
import os
from tqdm import tqdm
from scipy.spatial.transform import Rotation as R

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.transform import (
    align_two_point_sets, align_direction_and_translation,
    compute_plane_leveling_rotation, compute_z_flip_rotation,
)
from src.io_utils import (
    get_stage_files, get_output_dir, load_csv, save_csv,
    apply_transform_to_df, get_four_end_markers,
)
from src.markers import TARGET_DATES

FOUR_PT_DATASETS = ['240828', '240523']
TWO_PT_DATASETS = ['211215', '220819', '221112']

# Datasets whose LEnd/REnd labels are swapped relative to the player's hand
# labels. For these, the left wrist is physically near the REnd marker (not
# LEnd), which after alignment puts the body behind the keyboard. Swapping end
# marker labels before alignment fixes both the mirror and the 4-point plane
# geometry (the aux LEnd1/REnd1 are labeled consistently with the ends).
END_LABEL_SWAP_DATASETS = {'240523'}


def swap_end_labels(df):
    """Swap LEnd↔REnd and LEnd1↔REnd1 columns (keeps other L/R columns intact)."""
    rename = {}
    for prefix in ('LEnd', 'REnd', 'LEnd1', 'REnd1'):
        paired = {'LEnd': 'REnd', 'REnd': 'LEnd',
                  'LEnd1': 'REnd1', 'REnd1': 'LEnd1'}[prefix]
        for a in 'xyz':
            c = f'{prefix}_{a}'
            if c in df.columns:
                rename[c] = f'{paired}_{a}'
    return df.rename(columns=rename)


def load_reference_params():
    ref_path = os.path.join(get_output_dir('Aligned'), 'reference_params.npz')
    if not os.path.exists(ref_path):
        raise FileNotFoundError(f"Reference params not found: {ref_path}")
    params = np.load(ref_path)
    return {k: params[k] for k in params.files}


def get_file_median_endpoints(df):
    lend = df[['LEnd_x', 'LEnd_y', 'LEnd_z']].dropna().median().values
    rend = df[['REnd_x', 'REnd_y', 'REnd_z']].dropna().median().values
    return lend, rend


def reference_scale_ratio(df, ref_lend, ref_rend):
    """Reference / file LEnd-REnd distance. Diagnostic only; never applied.

    Motion capture is metric (mm), and the LEnd-REnd distance differs across
    cohorts because of where the end markers were placed on the keyboard
    (e.g. ~1205 mm on 220819/221112 vs ~1266 mm on 240523/240828), not
    because of a measurement scale error. Scaling the whole skeleton to the
    reference would inflate performers by the same ratio (up to 7.4%) and
    misplace fingertips against Stage 08's fixed physical key widths, whose
    keyboard frame is rebuilt per file from its own end markers.
    """
    al_lend, al_rend = get_file_median_endpoints(df)
    al_dist = np.linalg.norm(al_rend - al_lend)
    if al_dist < 1e-6:
        return 1.0
    return float(np.linalg.norm(ref_rend - ref_lend) / al_dist)


def compute_alignment_metrics(df, ref_lend, ref_rend):
    al_lend, al_rend = get_file_median_endpoints(df)
    ref_mid = (ref_lend + ref_rend) / 2
    al_mid = (al_lend + al_rend) / 2
    mid_err = np.linalg.norm(al_mid - ref_mid)

    al_vec = al_rend - al_lend
    ref_vec = ref_rend - ref_lend
    cos_a = np.clip(np.dot(al_vec, ref_vec) /
                     (np.linalg.norm(al_vec) * np.linalg.norm(ref_vec)), -1, 1)
    angle_err = np.degrees(np.arccos(cos_a))
    z_diff = abs(al_lend[2] - al_rend[2])
    return mid_err, angle_err, z_diff


def _wrist_or_elbow_median_3d(df):
    """Return a single 3D point: mean of L/R wrist (or elbow fallback) medians."""
    for joints in (['LWrist', 'RWrist'], ['LElbow', 'RElbow']):
        pts = []
        for j in joints:
            cols = [f'{j}_x', f'{j}_y', f'{j}_z']
            if all(c in df.columns for c in cols):
                m = df[cols].median().values
                if not np.isnan(m).any():
                    pts.append(m)
        if pts:
            return np.mean(pts, axis=0)
    return None


def ensure_canonical_orientation(df):
    """Snap the roll-around-keyboard-axis to put wrists in (+depth, +height) quadrant.

    After all prior alignment steps, the keyboard axis is defined by LEnd→REnd.
    The remaining orientation ambiguity is a rotation around this axis. We
    decompose the wrist position into (depth, height) components perpendicular
    to the keyboard axis, and rotate by 0/±90°/180° around the keyboard axis
    until both components are positive.

    Because the rotation axis *is* the keyboard axis, LEnd and REnd stay in
    place (their perpendicular components are zero), so no L/R label swap is
    needed and keyboard alignment to reference is preserved.
    """
    lend = df[['LEnd_x', 'LEnd_y', 'LEnd_z']].median().values
    rend = df[['REnd_x', 'REnd_y', 'REnd_z']].median().values
    if np.isnan(lend).any() or np.isnan(rend).any():
        return df, 0.0

    k_axis = rend - lend
    k_len = np.linalg.norm(k_axis)
    if k_len < 1e-6:
        return df, 0.0
    k_axis = k_axis / k_len

    # Build a height/depth basis orthogonal to keyboard axis.
    # height ≈ projection of world +Z onto perpendicular plane
    z_world = np.array([0.0, 0.0, 1.0])
    h_axis = z_world - np.dot(z_world, k_axis) * k_axis
    h_norm = np.linalg.norm(h_axis)
    if h_norm < 1e-6:
        # keyboard axis is parallel to Z (degenerate, shouldn't happen)
        h_axis = np.array([0.0, 0.0, 1.0])
    else:
        h_axis = h_axis / h_norm
    d_axis = np.cross(k_axis, h_axis)  # right-handed → "depth forward" if h is up

    # Wrist position relative to keyboard midpoint
    wrist = _wrist_or_elbow_median_3d(df)
    if wrist is None:
        return df, 0.0
    kbd_mid = (lend + rend) / 2
    rel = wrist - kbd_mid
    d_comp = float(np.dot(rel, d_axis))
    h_comp = float(np.dot(rel, h_axis))

    if abs(d_comp) < 1e-6 and abs(h_comp) < 1e-6:
        return df, 0.0

    # Find rotation angle (0, +90, 180, -90) that puts (d, h) into Q1 (both > 0)
    current_angle = np.arctan2(h_comp, d_comp)  # angle from +d axis, CCW around k
    best_snap_deg = 0
    for snap in (0, 90, 180, -90):
        new = current_angle + np.radians(snap)
        new = ((new + np.pi) % (2 * np.pi)) - np.pi  # wrap to [-pi, pi]
        deg = np.degrees(new)
        # First quadrant (strictly): 0 < deg < 90
        if 0 <= deg <= 90:
            best_snap_deg = snap
            break

    if best_snap_deg == 0:
        return df, 0.0

    # {d_axis, h_axis, k_axis} = k×h, h, k is left-handed, so a CCW rotation
    # by θ in the (d, h) plane corresponds to rotation by -θ around k_axis.
    R_roll = R.from_rotvec(-np.radians(best_snap_deg) * k_axis).as_matrix()
    # Rotate around keyboard midpoint, not origin, to keep LEnd/REnd exactly in place
    # (they may have tiny numerical offsets from the rotation axis).
    df_out = apply_transform_to_df(df, R_roll, t=kbd_mid - R_roll @ kbd_mid)
    return df_out, float(best_snap_deg)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Analytical roll correction for 2-point datasets
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def fix_roll_analytical(df):
    """
    Fix roll (rotation around keyboard axis) using elbow-wrist anatomical prior.

    After direction-only alignment, keyboard axis ≈ X.
    The Y-Z plane contains both HEIGHT and DEPTH of the arm.

    Key anatomical constraints:
    - Elbow is above wrist (height component, small: ~10-60mm)
    - Arm extends forward from keyboard (depth component, large: ~150-250mm)
    - HEIGHT should map to +Z, DEPTH should map to Y

    Algorithm:
    1. Compute elbow-wrist vector, project perpendicular to keyboard axis
    2. The SMALLER component (Y or Z) is HEIGHT, the LARGER is DEPTH
    3. Rotate so that HEIGHT component → +Z
    """
    lend, rend = get_file_median_endpoints(df)
    k_axis = rend - lend
    k_axis = k_axis / np.linalg.norm(k_axis)

    # Compute elbow-wrist "up" vectors
    arm_vecs = []
    for side in ['L', 'R']:
        ec = [f'{side}Elbow_{a}' for a in 'xyz']
        wc = [f'{side}Wrist_{a}' for a in 'xyz']
        if all(c in df.columns for c in ec + wc):
            e = df[ec].dropna().median().values
            w = df[wc].dropna().median().values
            if not np.isnan(e).any() and not np.isnan(w).any():
                arm_vecs.append(e - w)

    if not arm_vecs:
        return df, 0.0

    arm_up = np.mean(arm_vecs, axis=0)

    # Project perpendicular to keyboard axis
    arm_perp = arm_up - np.dot(arm_up, k_axis) * k_axis
    if np.linalg.norm(arm_perp) < 1e-6:
        return df, 0.0

    # Decompose arm_perp into Y and Z components
    # The SMALLER magnitude component is HEIGHT (elbow ~10-60mm above wrist)
    # The LARGER magnitude component is DEPTH (forearm ~150-250mm forward)
    y_comp = arm_perp[1]
    z_comp = arm_perp[2]

    if abs(y_comp) < abs(z_comp):
        # Height is currently in Y, depth in Z → need to rotate ~90°
        # We want the Y component (height) to go to +Z
        # Rotation angle: from current arm_perp to target direction
        # Target: height in +Z → target_perp direction has Z > 0
        # The height component (y_comp) should become +Z
        if y_comp >= 0:
            roll_angle = np.pi / 2   # rotate +90° to move +Y → +Z
        else:
            roll_angle = -np.pi / 2  # rotate -90° to move -Y → +Z
    else:
        # Height is already in Z, depth in Y → check sign only
        if z_comp >= 0:
            roll_angle = 0.0  # Already correct
        else:
            roll_angle = np.pi  # Flip 180°

    if abs(roll_angle) < 0.01:
        return df, 0.0

    R_roll = R.from_rotvec(roll_angle * k_axis).as_matrix()
    df_corrected = apply_transform_to_df(df, R_roll)

    return df_corrected, np.degrees(roll_angle)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Main alignment functions
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def align_4pt_dataset(dataset_id, ref_params):
    """4-point dataset: align to 240809, then self-level."""
    print(f"\n  [4pt] Aligning {dataset_id}...")

    input_files = get_stage_files('Centered', dataset_id)
    output_dir = get_output_dir('Aligned', dataset_id)
    ref_lend, ref_rend = ref_params['ref_lend'], ref_params['ref_rend']

    if not input_files:
        print(f"    No files found")
        return

    results = []
    for path in tqdm(input_files, desc=f"  {dataset_id}"):
        filename = os.path.basename(path)
        df = load_csv(path)

        required = ['LEnd_x', 'LEnd_y', 'LEnd_z', 'REnd_x', 'REnd_y', 'REnd_z']
        if not all(c in df.columns for c in required):
            save_csv(df, os.path.join(output_dir, filename))
            continue

        if dataset_id in END_LABEL_SWAP_DATASETS:
            df = swap_end_labels(df)

        src_lend, src_rend = get_file_median_endpoints(df)
        if np.isnan(src_lend).any() or np.isnan(src_rend).any():
            save_csv(df, os.path.join(output_dir, filename))
            continue

        # A: Cross-align to 240809
        R_al, t_al = align_two_point_sets(src_lend, src_rend, ref_lend, ref_rend)
        df_aligned = apply_transform_to_df(df, R_al, t_al)

        # B: Self-level using own 4 end markers
        mid_idx = len(df_aligned) // 2
        pts, valid = get_four_end_markers(df_aligned, mid_idx)
        if valid and len(pts) >= 3:
            R_level = compute_plane_leveling_rotation(pts)
            df_aligned = apply_transform_to_df(df_aligned, R_level)

        # C: Canonical orientation (snap roll so wrists are in +depth/+height quadrant)
        df_aligned, snap_deg = ensure_canonical_orientation(df_aligned)

        # D: No scaling -- see reference_scale_ratio for why.
        scale = reference_scale_ratio(df_aligned, ref_lend, ref_rend)

        mid_err, angle_err, z_diff = compute_alignment_metrics(df_aligned, ref_lend, ref_rend)
        pts_final, _ = get_four_end_markers(df_aligned, mid_idx)
        z_std_4pt = np.std(pts_final[:, 2]) if len(pts_final) >= 3 else float('inf')

        results.append((filename, mid_err, angle_err, z_diff, z_std_4pt))
        save_csv(df_aligned, os.path.join(output_dir, filename))
        if snap_deg != 0:
            print(f"    SNAP {snap_deg:+.0f}°: {filename}")

    if results:
        print(f"\n    {'File':<35} {'Mid':>6} {'Ang°':>7} {'Zdiff':>7} {'4ptZ':>7}")
        print(f"    {'-'*64}")
        for fn, me, ae, zd, zs in sorted(results, key=lambda x: -x[4]):
            s = 'O' if zs < 0.5 else '!'
            print(f"    {s} {fn:<33} {me:>6.3f} {ae:>7.4f} {zd:>7.3f} {zs:>7.3f}")


def align_2pt_dataset(dataset_id, ref_params):
    """2-point dataset: align + analytical roll correction."""
    print(f"\n  [2pt] Aligning {dataset_id}...")

    input_files = get_stage_files('Centered', dataset_id)
    output_dir = get_output_dir('Aligned', dataset_id)
    ref_lend, ref_rend = ref_params['ref_lend'], ref_params['ref_rend']

    if not input_files:
        print(f"    No files found")
        return

    results = []
    for path in tqdm(input_files, desc=f"  {dataset_id}"):
        filename = os.path.basename(path)
        df = load_csv(path)

        required = ['LEnd_x', 'LEnd_y', 'LEnd_z', 'REnd_x', 'REnd_y', 'REnd_z']
        if not all(c in df.columns for c in required):
            save_csv(df, os.path.join(output_dir, filename))
            continue

        src_lend, src_rend = get_file_median_endpoints(df)
        if np.isnan(src_lend).any() or np.isnan(src_rend).any():
            save_csv(df, os.path.join(output_dir, filename))
            continue

        # A: Direction + translation ONLY (no roll heuristic)
        R_al, t_al = align_direction_and_translation(src_lend, src_rend, ref_lend, ref_rend)
        df_aligned = apply_transform_to_df(df, R_al, t_al)

        # B: Analytical roll correction (elbow-wrist based)
        df_aligned, roll_deg = fix_roll_analytical(df_aligned)

        # C: Canonical orientation (snap roll so wrists are in +depth/+height quadrant)
        df_aligned, snap_deg = ensure_canonical_orientation(df_aligned)

        # D: No scaling -- see reference_scale_ratio for why.
        scale = reference_scale_ratio(df_aligned, ref_lend, ref_rend)

        mid_err, angle_err, z_diff = compute_alignment_metrics(df_aligned, ref_lend, ref_rend)

        # Verify: Elbow Z should be ~10-60mm above wrist (not 200mm = depth)
        ew_z = []
        for side in ['L', 'R']:
            ec, wc = f'{side}Elbow_z', f'{side}Wrist_z'
            if ec in df_aligned.columns and wc in df_aligned.columns:
                ew_z.append(df_aligned[ec].median() - df_aligned[wc].median())
        ew_mean = np.mean(ew_z) if ew_z else np.nan

        snap_tag = f"{snap_deg:+.0f}" if snap_deg != 0 else ""
        results.append((filename, mid_err, angle_err, z_diff, roll_deg, ew_mean, snap_tag))
        save_csv(df_aligned, os.path.join(output_dir, filename))

    if results:
        print(f"\n    {'File':<35} {'Mid':>5} {'Ang°':>7} {'Zdiff':>6} {'Roll°':>7} {'E-W_z':>6} {'Flip':>5}")
        print(f"    {'-'*73}")
        for fn, me, ae, zd, rd, ew, fl in sorted(results, key=lambda x: -abs(x[4])):
            s = 'O' if (not np.isnan(ew) and 5 < ew < 80) else '?'
            print(f"    {s} {fn:<33} {me:>5.3f} {ae:>7.4f} {zd:>6.3f} {rd:>+7.1f} {ew:>6.1f} {fl:>5}")
        avg_z = np.mean([r[3] for r in results])
        print(f"    Mean Z-diff: {avg_z:.3f}")


def run():
    print("\n" + "=" * 60)
    print("  STEP 5: Cross-Year Alignment → 240809")
    print("=" * 60)

    ref_params = load_reference_params()

    # Phase 1: 4-point datasets
    for ds in FOUR_PT_DATASETS:
        if ds in TARGET_DATES:
            align_4pt_dataset(ds, ref_params)

    # Phase 2: 2-point datasets with analytical roll
    for ds in TWO_PT_DATASETS:
        if ds in TARGET_DATES:
            align_2pt_dataset(ds, ref_params)

    print("\n  Step 5 complete.")


if __name__ == '__main__':
    run()
