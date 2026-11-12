"""
File I/O utilities for the alignment pipeline.
"""

import os
import glob
import pandas as pd
import numpy as np


def get_data_root():
    """Get the data root directory."""
    # Try common locations
    for path in ['Data/Processed', './Data/Processed']:
        if os.path.exists(path):
            return path
    raise FileNotFoundError("Data root not found. Expected Data/Processed")


def get_raw_files(dataset_id, config):
    """Get list of raw CSV file paths for a dataset."""
    data_root = get_data_root()
    raw_dir = os.path.join(data_root, config['raw_path'])
    pattern = os.path.join(raw_dir, config['raw_glob'])
    files = sorted(glob.glob(pattern))
    if not files:
        print(f"  WARNING: No files found at {pattern}")
    return files


def get_output_dir(stage_name, dataset_id=None):
    """
    Get output directory for a pipeline stage.

    Output structure:
        Data/Pipeline/02_spatial/hand/{stage_name}/{dataset_id}/*.csv
    """
    base = 'Data/Pipeline/02_spatial/hand'
    if dataset_id:
        path = os.path.join(base, stage_name, dataset_id)
    else:
        path = os.path.join(base, stage_name)
    os.makedirs(path, exist_ok=True)
    return path


def get_stage_files(stage_name, dataset_id):
    """Get list of CSV files from a pipeline stage output."""
    stage_dir = get_output_dir(stage_name, dataset_id)
    return sorted(glob.glob(os.path.join(stage_dir, '*.csv')))


def load_csv(path, has_index=None):
    """
    Load a CSV file, auto-detecting index column.

    Args:
        path: file path
        has_index: force index column handling (None=auto)
    """
    df = pd.read_csv(path)

    # Auto-detect: if first column looks like an index, handle it
    if has_index is None:
        first_col = df.columns[0]
        if first_col.startswith('Unnamed') or first_col == '':
            df = df.drop(columns=[first_col])

    return df


def save_csv(df, path, index=False):
    """Save DataFrame to CSV."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=index)


def get_end_marker_positions(df, frame_idx=None):
    """
    Extract LEnd and REnd positions from a DataFrame.

    Args:
        df: DataFrame with LEnd_x/y/z, REnd_x/y/z columns
        frame_idx: specific frame index (None = all frames)

    Returns:
        lend: (3,) or (N, 3) array
        rend: (3,) or (N, 3) array
    """
    if frame_idx is not None:
        row = df.iloc[frame_idx]
        lend = np.array([row['LEnd_x'], row['LEnd_y'], row['LEnd_z']])
        rend = np.array([row['REnd_x'], row['REnd_y'], row['REnd_z']])
    else:
        lend = df[['LEnd_x', 'LEnd_y', 'LEnd_z']].values
        rend = df[['REnd_x', 'REnd_y', 'REnd_z']].values
    return lend, rend


def get_four_end_markers(df, frame_idx):
    """
    Extract end marker positions for 4-point leveling.
    Returns whatever valid markers are available (3 or 4).

    Args:
        df: DataFrame with LEnd1_x/y/z, LEnd_x/y/z, REnd1_x/y/z, REnd_x/y/z
        frame_idx: frame index

    Returns:
        points: (N, 3) array of valid end marker positions
        valid: bool indicating >= 3 markers available (enough for plane fitting)
    """
    row = df.iloc[frame_idx]
    markers = [
        ('LEnd1_x', 'LEnd1_y', 'LEnd1_z'),
        ('LEnd_x', 'LEnd_y', 'LEnd_z'),
        ('REnd1_x', 'REnd1_y', 'REnd1_z'),
        ('REnd_x', 'REnd_y', 'REnd_z'),
    ]

    points = []
    for mx, my, mz in markers:
        if mx in row.index and my in row.index and mz in row.index:
            x, y, z = row[mx], row[my], row[mz]
            if not (np.isnan(x) or np.isnan(y) or np.isnan(z)):
                points.append([x, y, z])

    if len(points) >= 3:
        return np.array(points), True
    return np.array(points) if points else np.empty((0, 3)), False


def apply_transform_to_df(df, R_mat, t=None):
    """
    Apply rigid transformation to all coordinate columns in a DataFrame.

    Args:
        df: DataFrame with *_x, *_y, *_z columns
        R_mat: (3, 3) rotation matrix
        t: (3,) translation vector (optional)

    Returns:
        df_out: transformed DataFrame (copy)
    """
    from .markers import get_marker_names

    df_out = df.copy()
    markers = get_marker_names(df)

    for marker in markers:
        cx, cy, cz = f'{marker}_x', f'{marker}_y', f'{marker}_z'
        if cx in df.columns and cy in df.columns and cz in df.columns:
            coords = df[[cx, cy, cz]].values  # (N, 3)
            valid = ~np.isnan(coords).any(axis=1)

            if valid.any():
                valid_coords = coords[valid]
                transformed = (R_mat @ valid_coords.T).T
                if t is not None:
                    transformed += t

                df_out.loc[df_out.index[valid], cx] = transformed[:, 0]
                df_out.loc[df_out.index[valid], cy] = transformed[:, 1]
                df_out.loc[df_out.index[valid], cz] = transformed[:, 2]

    return df_out


def apply_per_frame_transform_to_df(df, rotations, translations=None):
    """
    Apply per-frame rigid transformations to all coordinate columns.
    Vectorized: stacks all rotations into (N, 3, 3) and uses einsum.

    Args:
        df: DataFrame
        rotations: list of (3, 3) matrices, one per frame
        translations: list of (3,) vectors (optional)

    Returns:
        df_out: transformed DataFrame
    """
    from .markers import get_marker_names

    df_out = df.copy()
    markers = get_marker_names(df)
    n_frames = len(df)
    R_stack = np.array(rotations)  # (N, 3, 3)

    for marker in markers:
        cx, cy, cz = f'{marker}_x', f'{marker}_y', f'{marker}_z'
        if cx not in df.columns or cy not in df.columns or cz not in df.columns:
            continue

        coords = df_out[[cx, cy, cz]].values  # (N, 3)
        valid = ~np.isnan(coords).any(axis=1)

        if valid.any():
            # einsum: for each frame i, rotated[i] = R[i] @ coords[i]
            valid_idx = np.where(valid)[0]
            valid_coords = coords[valid_idx]  # (M, 3)
            valid_R = R_stack[valid_idx]  # (M, 3, 3)
            rotated = np.einsum('ijk,ik->ij', valid_R, valid_coords)  # (M, 3)

            if translations is not None:
                t_stack = np.array(translations)
                rotated += t_stack[valid_idx]

            coords[valid_idx] = rotated
            df_out[cx] = coords[:, 0]
            df_out[cy] = coords[:, 1]
            df_out[cz] = coords[:, 2]

    return df_out
