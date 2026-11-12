"""
Body motion data I/O: parse TSV (Qualisys) and CSV formats.
"""
import os
import math
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Tuple


def parse_qualisys_tsv(filepath: str) -> Tuple[float, pd.DataFrame]:
    """Parse Qualisys TSV body motion file (240828, 221112 format).

    Header block:
        NO_OF_FRAMES  N
        FREQUENCY     F
        MARKER_NAMES  M1  M2  ...
        Frame  Time  M1 X  M1 Y  M1 Z  M2 X ...

    Returns:
        fps: float
        df: DataFrame with columns like HEADT_x, HEADT_y, HEADT_z, ...
    """
    with open(filepath, 'r', encoding='utf-8', errors='replace') as f:
        lines = f.readlines()

    fps = 120.0
    marker_names = []
    data_start = None
    has_frame_time = False

    for i, line in enumerate(lines[:20]):
        parts = line.strip().split('\t')
        key = parts[0].strip()
        if key == 'FREQUENCY':
            fps = float(parts[1].strip())
        elif key == 'MARKER_NAMES':
            marker_names = [p.strip() for p in parts[1:] if p.strip()]
        elif key == 'TRAJECTORY_TYPES':
            # Data starts on next line (no Frame/Time header in some datasets)
            if data_start is None:
                data_start = i + 1
        elif key == 'Frame':
            # Explicit Frame/Time header row ??data starts after
            data_start = i + 1
            has_frame_time = True
            break

    if data_start is None or not marker_names:
        raise ValueError(f"Cannot parse TSV header: {filepath}")

    # Build column names
    columns = []
    if has_frame_time:
        columns = ['Frame', 'Time']
    for m in marker_names:
        columns.extend([f'{m}_x', f'{m}_y', f'{m}_z'])

    # Parse data rows
    rows = []
    for line in lines[data_start:]:
        parts = line.strip().split('\t')
        if not parts or not parts[0].strip():
            continue
        row = []
        for p in parts:
            p = p.strip()
            try:
                row.append(float(p))
            except ValueError:
                row.append(np.nan)
        rows.append(row)

    if not rows:
        raise ValueError(f"No data rows: {filepath}")

    # Align column count
    n_cols = len(rows[0])
    if len(columns) > n_cols:
        columns = columns[:n_cols]
    elif len(columns) < n_cols:
        columns.extend([f'_extra_{i}' for i in range(n_cols - len(columns))])

    df = pd.DataFrame(rows, columns=columns)
    df = df.drop(columns=[c for c in df.columns if c in ('Frame', 'Time') or c.startswith('_extra')], errors='ignore')

    return fps, df


def parse_processed_body_csv(filepath: str) -> Tuple[float, pd.DataFrame]:
    """Parse 240809 processed body CSV (space-separated axes: 'HEADT X').

    Returns:
        fps: float (default 120)
        df: DataFrame with columns like HEADT_x, HEADT_y, HEADT_z, ...
    """
    df_raw = pd.read_csv(filepath)

    # Rename "MARKER X" ??"MARKER_x" etc.
    rename = {}
    for col in df_raw.columns:
        parts = col.rsplit(' ', 1)
        if len(parts) == 2 and parts[1] in ('X', 'Y', 'Z'):
            rename[col] = f'{parts[0]}_{parts[1].lower()}'

    df = df_raw.rename(columns=rename)
    df = df.drop(columns=['Time'], errors='ignore')
    # Drop any non-marker columns
    df = df[[c for c in df.columns if c.endswith(('_x', '_y', '_z'))]]

    return 120.0, df


def load_body_file(filepath: str) -> Tuple[float, pd.DataFrame]:
    """Load body motion file, auto-detecting format."""
    ext = os.path.splitext(filepath)[1].lower()
    if ext == '.tsv':
        return parse_qualisys_tsv(filepath)
    elif ext == '.csv':
        return parse_processed_body_csv(filepath)
    else:
        raise ValueError(f"Unsupported body format: {ext}")


def get_body_files(dataset_id: str) -> Dict[str, str]:
    """Return {trial_name: filepath} for body data of a dataset.

    Reads from Stage 01 canonical output so trial_name is the canonical stem
    (e.g. ``220819_C0_N0_T1_P1_t1``). Downstream pairing against Stage 02 hand
    therefore joins by identical canonical stems.
    """
    paths_map = {
        '220819': [f'Data/Pipeline/01_ingest/Normalized/220819/{s}/motion_body' for s in ['C0_N0','C0_N1','C0_N2','C1_N0','C1_N1']],
        '240523': ['Data/Pipeline/01_ingest/Normalized/240523/C1_N0/motion_body'],
        '240809': ['Data/Pipeline/01_ingest/Normalized/240809/C0_N0/motion_body'],
        '240828': [f'Data/Pipeline/01_ingest/Normalized/240828/C0_N{n}/motion_body' for n in range(4, 9)],
        '221112': [f'Data/Pipeline/01_ingest/Normalized/221112/C1_N{n}/motion_body' for n in range(2, 12)],
    }
    if dataset_id not in paths_map:
        return {}

    result = {}
    for dirpath in paths_map[dataset_id]:
        if not os.path.isdir(dirpath):
            continue
        for f in sorted(os.listdir(dirpath)):
            if f.endswith(('.csv', '.tsv')):
                name = os.path.splitext(f)[0]
                result[name] = os.path.join(dirpath, f)
    return result
