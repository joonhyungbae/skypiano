"""
Step 1: Column Unification + FPS Conversion

- Parse Vicon C3D CSV headers for raw datasets (240828, 221112, 240523)
- Rename raw columns to unified naming convention
- Convert 240809's ./.1/.2 notation to _z/_x/_y
- Interpolate 100fps → 120fps where needed
- Drop unnecessary columns (Time, TimeCode, index cols)

Input:  Raw CSVs from each dataset
Output: Data/Pipeline/02_spatial/hand/Unified/{dataset_id}/*.csv
"""

import pandas as pd
import numpy as np
import os
import glob
from tqdm import tqdm

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.markers import DATASET_CONFIGS, apply_wrist_center, filter_to_stage2_columns
from src.io_utils import get_output_dir, save_csv


def get_data_root():
    """Get the data root directory."""
    for path in ['Data/Processed', 'Data']:
        if os.path.exists(path):
            return 'Data'
    raise FileNotFoundError("Data root not found")


def get_raw_files_for_config(dataset_id, config):
    """Get raw file list, handling single path or list of paths."""
    data_root = get_data_root()
    raw_paths = config['raw_path']

    # Normalize to list
    if isinstance(raw_paths, str):
        raw_paths = [raw_paths]

    all_files = []
    for rp in raw_paths:
        # Try under Processed/ first, then directly under data_root
        for base in [os.path.join(data_root, 'Processed'), data_root]:
            full_dir = os.path.join(base, rp)
            if os.path.isdir(full_dir):
                pattern = os.path.join(full_dir, config['raw_glob'])
                all_files.extend(sorted(glob.glob(pattern)))
                break
        else:
            # A missing input dir must be visible: without it the downstream
            # steps silently reprocess whatever Unified/ already holds.
            print(f"  WARNING: input dir not found for {dataset_id}: {rp}")

    if not all_files:
        print(f"  WARNING: No files found for {dataset_id}")
    return all_files


def convert_dot_notation(df):
    """
    Convert 240809's column naming from ./.1/.2 notation to _z/_x/_y.
    """
    renamed = {}
    for col in df.columns:
        if col.endswith('.1'):
            renamed[col] = col[:-2] + '_x'
        elif col.endswith('.2'):
            renamed[col] = col[:-2] + '_y'
        elif col not in ('Time', 'TimeCode') and '.' not in col:
            renamed[col] = col + '_z'
        else:
            renamed[col] = col
    return df.rename(columns=renamed)


def reorder_axes(df, axis_order):
    """Relabel parsed x/y/z columns: raw axis i becomes axis_order[i]."""
    mapping = dict(zip('xyz', axis_order))
    renamed = {}
    for col in df.columns:
        if len(col) > 2 and col[-2] == '_' and col[-1] in mapping:
            renamed[col] = col[:-1] + mapping[col[-1]]
    return df.rename(columns=renamed)


def parse_metadata_header_csv(filepath):
    """
    Parse a Vicon CSV export with multi-line metadata header.

    Header structure (v1.23 / v1.24):
      Line 0: Format Version, 1.24, Take Name, ..., Capture Frame Rate, 119.88, ...
      Line 1: ,Type,,Marker,Marker,...       (or ,Type,Marker,Marker,...)
      Line 2: ,Name,,Markerset 001:LID,...   (or ,Name,MarkerSet 001:LEND,...)
      Line 3: ,ID,,...                        (or ,Component,,X,Y,Z,...)
      Line 4: (may be Position row or Frame,Time row)
      Data rows follow

    Returns:
        df: DataFrame with columns like LID_x, LID_y, LID_z, ...
        fps: float extracted from header
    """
    # Read all lines
    with open(filepath, 'r', encoding='utf-8', errors='replace') as f:
        lines = f.readlines()

    # Line 0: metadata → extract FPS
    meta_parts = lines[0].strip().split(',')
    fps = 120.0
    for i, part in enumerate(meta_parts):
        if 'Capture Frame Rate' in part and i + 1 < len(meta_parts):
            try:
                fps = float(meta_parts[i + 1])
            except ValueError:
                pass
            break

    # Find the Name row (contains marker names like "Markerset 001:LID")
    name_row_idx = None
    for i in range(1, min(6, len(lines))):
        if ',Name,' in lines[i] or lines[i].startswith(',Name,'):
            name_row_idx = i
            break

    if name_row_idx is None:
        raise ValueError(f"Cannot find Name row in {filepath}")

    name_parts = lines[name_row_idx].strip().split(',')

    # Extract marker names, stripping prefix like "Markerset 001:" or "MarkerSet 001:"
    marker_names = []
    for part in name_parts:
        part = part.strip().strip('"')
        if ':' in part:
            marker_names.append(part.split(':')[-1])
        else:
            marker_names.append(part)

    # Find data start row (first row starting with a digit = frame number)
    data_start = None
    for i in range(name_row_idx + 1, min(name_row_idx + 5, len(lines))):
        stripped = lines[i].strip()
        if stripped and stripped[0].isdigit():
            data_start = i
            break

    if data_start is None:
        raise ValueError(f"Cannot find data start in {filepath}")

    # The first columns are Frame, Time (Seconds), [TimeCode]
    # Then marker data in groups of 3 (X, Y, Z)
    # Determine how many prefix columns before marker data
    # By comparing name_parts: first entries before marker names are empty or labels
    prefix_count = 0
    for part in name_parts:
        part = part.strip().strip('"')
        if ':' in part:
            break
        prefix_count += 1

    # Build column names: prefix cols + marker_x, marker_y, marker_z
    # Group marker names into triplets
    marker_entries = marker_names[prefix_count:]
    columns = ['Frame', 'Time']
    if prefix_count > 2:
        columns.append('TimeCode')

    # Each marker appears 3 times in sequence (X, Y, Z). If the same marker
    # name shows up multiple times (happens when a mocap scene has several
    # copies of the same labelled marker, e.g. 4×LEND), suffix the extras so
    # downstream dedupe can pick one deterministically.
    i = 0
    seen_markers: dict[str, int] = {}
    while i < len(marker_entries):
        name = marker_entries[i]
        if name:
            count = seen_markers.get(name, 0)
            tag = name if count == 0 else f"{name}__dup{count}"
            seen_markers[name] = count + 1
            columns.append(f'{tag}_x')
            columns.append(f'{tag}_y')
            columns.append(f'{tag}_z')
            i += 3
        else:
            i += 1

    # Identify which columns should stay string (TimeCode) vs numeric
    tc_idx = columns.index('TimeCode') if 'TimeCode' in columns else -1

    # Parse data rows
    data_lines = lines[data_start:]
    rows = []
    for line in data_lines:
        parts = line.strip().split(',')
        if not parts or not parts[0].strip().replace('.','').isdigit():
            continue
        row = []
        for i, p in enumerate(parts):
            p = p.strip().strip('"')
            if i == tc_idx:
                # Preserve TimeCode as string
                row.append(p if p else None)
                continue
            try:
                row.append(float(p))
            except ValueError:
                row.append(np.nan)
        rows.append(row)

    if not rows:
        raise ValueError(f"No data rows found in {filepath}")

    # Align column count
    n_cols = len(rows[0])
    if len(columns) > n_cols:
        columns = columns[:n_cols]
    elif len(columns) < n_cols:
        columns.extend([f'_extra_{i}' for i in range(n_cols - len(columns))])

    df = pd.DataFrame(rows, columns=columns)

    # Drop Frame and Time, but KEEP TimeCode — Stage 03 midi_motion s03 uses it to re-extract
    # session MIDI aligned to Stage 02's possibly-trimmed frame range.
    for col in ['Frame', 'Time']:
        if col in df.columns:
            df = df.drop(columns=[col])

    # Drop _extra columns
    df = df[[c for c in df.columns if not c.startswith('_extra')]]

    return df, fps


def interpolate_fps(df, from_fps=100, to_fps=120):
    """Interpolate data from one FPS to another using linear interpolation."""
    original_time = np.arange(len(df)) / from_fps
    new_time = np.arange(0, original_time[-1], 1 / to_fps)

    # Use df.dtypes (Series) so duplicate column names don't return a DataFrame.
    dtypes = df.dtypes
    numeric_mask = dtypes.apply(lambda d: d.kind in "fiu")
    numeric_cols = list(dtypes.index[numeric_mask])
    # df[numeric_cols] may still return duplicated columns; dedupe by position.
    seen = set()
    unique_cols = []
    for c in numeric_cols:
        if c not in seen:
            unique_cols.append(c)
            seen.add(c)
    numeric_cols = unique_cols
    interpolated = pd.DataFrame(
        {col: np.interp(new_time, original_time, df[col].values) for col in numeric_cols},
    )
    return interpolated.reset_index(drop=True)


def process_dataset(dataset_id, config):
    """Process a single dataset: parse + rename columns + FPS conversion."""
    print(f"\n{'='*60}")
    print(f"  Processing {dataset_id}")
    print(f"{'='*60}")

    raw_files = get_raw_files_for_config(dataset_id, config)
    output_dir = get_output_dir('Unified', dataset_id)

    print(f"  Found {len(raw_files)} files")
    print(f"  Metadata header: {config.get('has_metadata_header', False)}")
    print(f"  Output: {output_dir}")

    has_meta_cfg = config.get('has_metadata_header', False)

    for path in tqdm(raw_files, desc=f"  {dataset_id}"):
        filename = os.path.basename(path)

        try:
            # Auto-detect metadata header per file
            if has_meta_cfg == 'auto':
                with open(path, 'r', encoding='utf-8', errors='replace') as peek:
                    first_line = peek.readline()
                has_meta = first_line.startswith('Format Version')
            else:
                has_meta = has_meta_cfg

            if has_meta:
                df, actual_fps = parse_metadata_header_csv(path)
                if config.get('axis_order'):
                    df = reorder_axes(df, config['axis_order'])
                # Use actual FPS for conversion decision
                if config['fps_conversion']:
                    df = interpolate_fps(df, from_fps=actual_fps, to_fps=120)
            else:
                df = pd.read_csv(path)

                # Drop unnecessary columns
                for col in config['columns_to_drop']:
                    if col in df.columns:
                        df = df.drop(columns=[col])

                # Remove unnamed index columns
                unnamed_cols = [c for c in df.columns if c.startswith('Unnamed')]
                if unnamed_cols:
                    df = df.drop(columns=unnamed_cols)

                # Convert ./.1/.2 notation for 240809
                if config.get('needs_dot_conversion', False):
                    df = convert_dot_notation(df)

                # FPS conversion
                if config['fps_conversion']:
                    df = interpolate_fps(df, from_fps=config['fps'], to_fps=120)

            # Apply file-specific column overrides (e.g., mislabeled markers)
            overrides = config.get('file_column_overrides', {})
            if filename in overrides:
                df = df.rename(columns=overrides[filename])

            # Derived wrist joint centre (before renaming, from raw marker names)
            df = apply_wrist_center(df, config)

            # Apply column mapping (common for all)
            df = df.rename(columns=config['column_mapping'])

            # Filter to canonical columns only (drops extras like ASIS, TRUNK, HMID, etc.)
            df = filter_to_stage2_columns(df)

            # Save
            out_path = os.path.join(output_dir, filename)
            save_csv(df, out_path)

        except Exception as e:
            print(f"  ERROR {filename}: {e}")

    print(f"  Done: {len(raw_files)} files -> {output_dir}")


def run():
    """Run Step 1 for all datasets."""
    print("\n" + "=" * 60)
    print("  STEP 1: Column Unification + FPS Conversion")
    print("=" * 60)

    for dataset_id, config in DATASET_CONFIGS.items():
        process_dataset(dataset_id, config)

    print("\n  Step 1 complete.")


if __name__ == '__main__':
    run()
