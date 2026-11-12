"""
3D transformation utilities for motion capture alignment.

Provides:
- SVD-based plane fitting
- Rotation matrix computation
- Per-frame and per-file alignment
- Procrustes-like 2-point alignment with roll correction
"""

import numpy as np
from scipy.spatial.transform import Rotation as R
from scipy.optimize import minimize


def fit_plane_normal(points):
    """
    Fit a plane to 3D points using SVD and return the normal vector.

    Args:
        points: (N, 3) array of 3D points

    Returns:
        normal: (3,) unit normal vector (oriented toward +Z)
    """
    centroid = np.mean(points, axis=0)
    centered = points - centroid
    _, S, Vt = np.linalg.svd(centered)
    normal = Vt[-1, :]
    if normal[2] < 0:
        normal = -normal
    return normal


def rotation_to_align_vectors(v_from, v_to):
    """
    Compute rotation matrix that rotates v_from to v_to.

    Args:
        v_from: source vector (3,)
        v_to: target vector (3,)

    Returns:
        R_mat: (3, 3) rotation matrix
    """
    v_from = v_from / np.linalg.norm(v_from)
    v_to = v_to / np.linalg.norm(v_to)

    axis = np.cross(v_from, v_to)
    axis_norm = np.linalg.norm(axis)

    if axis_norm < 1e-10:
        # Vectors are parallel
        if np.dot(v_from, v_to) > 0:
            return np.eye(3)
        else:
            # 180-degree rotation around any perpendicular axis
            perp = np.array([1, 0, 0]) if abs(v_from[0]) < 0.9 else np.array([0, 1, 0])
            axis = np.cross(v_from, perp)
            axis = axis / np.linalg.norm(axis)
            return R.from_rotvec(np.pi * axis).as_matrix()

    axis = axis / axis_norm
    angle = np.arccos(np.clip(np.dot(v_from, v_to), -1.0, 1.0))
    return R.from_rotvec(angle * axis).as_matrix()


def compute_plane_leveling_rotation(points):
    """
    Compute rotation that makes a set of points lie on a horizontal plane (Z=const).

    Uses SVD plane fitting + BFGS optimization for robustness.

    Args:
        points: (N, 3) array, N >= 3

    Returns:
        R_mat: (3, 3) rotation matrix
    """
    if len(points) < 3:
        return np.eye(3)

    # Step 1: SVD plane fitting
    normal = fit_plane_normal(points)
    target = np.array([0, 0, 1])
    R_svd = rotation_to_align_vectors(normal, target)

    # Check quality
    rotated = (R_svd @ points.T).T
    z_std = np.std(rotated[:, 2])

    if z_std < 0.01:
        return R_svd

    # Step 2: BFGS optimization starting from SVD solution
    initial_rotvec = R.from_matrix(R_svd).as_rotvec()

    def objective(rotvec):
        try:
            rot = R.from_rotvec(rotvec).as_matrix()
            rotated = (rot @ points.T).T
            return np.std(rotated[:, 2])
        except:
            return 1e6

    result = minimize(objective, initial_rotvec, method='BFGS',
                      options={'maxiter': 200, 'gtol': 1e-8})

    if result.success and result.fun < z_std:
        return R.from_rotvec(result.x).as_matrix()

    return R_svd


def compute_z_flip_rotation(lend, rend):
    """
    Compute a 180-degree rotation around the LEnd→REnd axis.
    Used to flip Z direction when hands end up below the keyboard.

    Args:
        lend: (3,) LEnd position
        rend: (3,) REnd position

    Returns:
        R_mat: (3, 3) rotation matrix (180° around keyboard axis)
    """
    axis = rend - lend
    axis = axis / np.linalg.norm(axis)
    return R.from_rotvec(np.pi * axis).as_matrix()


def compute_per_frame_rotation(points):
    """
    Compute leveling rotation for a single frame's end marker points.
    Uses SVD only (no BFGS) for speed in per-frame mode.

    Args:
        points: (N, 3) array of end marker positions

    Returns:
        R_mat: (3, 3) rotation matrix
    """
    if len(points) < 3:
        return np.eye(3)

    normal = fit_plane_normal(points)
    target = np.array([0, 0, 1])
    return rotation_to_align_vectors(normal, target)


def smooth_rotations(rotations, window=5):
    """
    Temporally smooth a sequence of rotation matrices using quaternion averaging.

    Args:
        rotations: list of (3, 3) rotation matrices
        window: smoothing window size

    Returns:
        smoothed: list of (3, 3) rotation matrices
    """
    n = len(rotations)
    smoothed = []

    for i in range(n):
        start = max(0, i - window // 2)
        end = min(n, i + window // 2 + 1)

        quats = [R.from_matrix(rot).as_quat() for rot in rotations[start:end]]

        # Ensure consistent quaternion sign (avoid antipodal averaging)
        ref_quat = quats[0]
        for j in range(1, len(quats)):
            if np.dot(quats[j], ref_quat) < 0:
                quats[j] = -quats[j]

        quat_mean = np.mean(quats, axis=0)
        quat_mean = quat_mean / np.linalg.norm(quat_mean)

        smoothed.append(R.from_quat(quat_mean).as_matrix())

    return smoothed


def apply_rigid_transform(coords, R_mat, t=None):
    """
    Apply rigid transformation (rotation + optional translation) to coordinates.

    Args:
        coords: (N, 3) array
        R_mat: (3, 3) rotation matrix
        t: (3,) translation vector (optional)

    Returns:
        transformed: (N, 3) array
    """
    transformed = (R_mat @ coords.T).T
    if t is not None:
        transformed += t
    return transformed


def align_direction_and_translation(src_L, src_R, ref_L, ref_R):
    """
    Align source to reference using only direction + translation (5 DOF).
    Does NOT attempt roll correction — leaves that for analytical fix.

    Returns:
        R_mat: (3, 3) rotation matrix (direction only)
        t: (3,) translation vector
    """
    src_mid = (src_L + src_R) / 2
    ref_mid = (ref_L + ref_R) / 2

    src_vec = src_R - src_L
    ref_vec = ref_R - ref_L
    src_unit = src_vec / np.linalg.norm(src_vec)
    ref_unit = ref_vec / np.linalg.norm(ref_vec)

    R_align = rotation_to_align_vectors(src_unit, ref_unit)
    t = ref_mid - R_align @ src_mid
    return R_align, t


def align_two_point_sets(src_L, src_R, ref_L, ref_R, ref_plane_normal=None):
    """
    Align source data to reference using 2 corresponding point pairs (LEnd, REnd).

    Solves the 6-DOF problem:
    - 3 DOF translation: match midpoints
    - 2 DOF rotation: align LEnd→REnd vectors
    - 1 DOF roll: use horizontal plane constraint

    Args:
        src_L: (3,) source LEnd position
        src_R: (3,) source REnd position
        ref_L: (3,) reference LEnd position
        ref_R: (3,) reference REnd position
        ref_plane_normal: (3,) reference piano plane normal (default: [0,0,1] if 240809 is leveled)

    Returns:
        R_mat: (3, 3) rotation matrix
        t: (3,) translation vector
        Such that: target = R_mat @ source + t
    """
    if ref_plane_normal is None:
        ref_plane_normal = np.array([0.0, 0.0, 1.0])

    # Step 1: Translation to match midpoints
    src_mid = (src_L + src_R) / 2
    ref_mid = (ref_L + ref_R) / 2

    # Step 2: Align LEnd→REnd vectors (2 rotational DOF)
    src_vec = src_R - src_L
    ref_vec = ref_R - ref_L

    src_vec_unit = src_vec / np.linalg.norm(src_vec)
    ref_vec_unit = ref_vec / np.linalg.norm(ref_vec)

    R_align = rotation_to_align_vectors(src_vec_unit, ref_vec_unit)

    # Step 3: Solve remaining roll DOF (rotation around ref_vec axis)
    # Constraint: the piano plane should be horizontal
    # After R_align, we need to find roll angle θ around ref_vec_unit such that
    # the resulting coordinate system has the piano plane normal ≈ [0,0,1]
    #
    # We construct a local frame and optimize θ to minimize the tilt.

    def roll_objective(theta):
        R_roll = R.from_rotvec(theta[0] * ref_vec_unit).as_matrix()
        R_total = R_roll @ R_align

        # Apply to source points
        src_L_rot = R_total @ (src_L - src_mid)
        src_R_rot = R_total @ (src_R - src_mid)

        # The LEnd-REnd line should be horizontal (same Z)
        z_diff = abs(src_L_rot[2] - src_R_rot[2])

        # Additionally, we want a "up" direction to align with +Z
        # Use the perpendicular direction in the source frame
        src_up = np.cross(src_vec_unit, np.array([0, 0, 1]))
        if np.linalg.norm(src_up) < 1e-6:
            src_up = np.cross(src_vec_unit, np.array([0, 1, 0]))
        src_up = src_up / np.linalg.norm(src_up)

        # After rotation, this should point roughly horizontal (Z ≈ 0)
        rotated_up = R_total @ src_up
        return abs(rotated_up[2]) + z_diff

    result = minimize(roll_objective, [0.0], method='Nelder-Mead',
                      options={'maxiter': 500, 'xatol': 1e-8})

    if result.success:
        R_roll = R.from_rotvec(result.x[0] * ref_vec_unit).as_matrix()
        R_total = R_roll @ R_align
    else:
        R_total = R_align

    # Step 4: Compute translation
    t = ref_mid - R_total @ src_mid

    # Step 5: Scale correction (match distances)
    src_dist = np.linalg.norm(src_vec)
    ref_dist = np.linalg.norm(ref_vec)
    scale = ref_dist / src_dist if src_dist > 1e-6 else 1.0

    # For rigid alignment we typically don't scale, but if distances are very different
    # it indicates a unit mismatch. Log a warning but don't scale.
    if abs(scale - 1.0) > 0.1:
        print(f"  WARNING: Distance mismatch (scale={scale:.3f}). "
              f"src={src_dist:.1f}, ref={ref_dist:.1f}")

    return R_total, t
