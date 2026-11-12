"""
Step 3: Center Coordinates

- Translate all coordinates so that the midpoint of LEnd and REnd is at the origin
- Applied per-frame to handle any drift

Input:  Data/Pipeline/02_spatial/hand/Unified/{dataset_id}/*.csv
Output: Data/Pipeline/02_spatial/hand/Centered/{dataset_id}/*.csv
"""

import pandas as pd
import numpy as np
import os
from tqdm import tqdm

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.markers import DATASET_CONFIGS, get_coordinate_columns
from src.io_utils import get_stage_files, get_output_dir, load_csv, save_csv


def center_to_end_midpoint(df):
    """
    Center all coordinates so that (LEnd + REnd) / 2 = origin, per frame.

    Args:
        df: DataFrame with LEnd_x/y/z, REnd_x/y/z and other coordinate columns

    Returns:
        df_centered: centered DataFrame
    """
    df = df.copy()

    center_x = (df['LEnd_x'] + df['REnd_x']) / 2
    center_y = (df['LEnd_y'] + df['REnd_y']) / 2
    center_z = (df['LEnd_z'] + df['REnd_z']) / 2

    # End markers are attached to the piano, so frames with NaN in LEnd or REnd
    # fall back to the file-wide median center (piano doesn't actually move).
    # Without this, NaN propagates into every column for those frames and makes
    # the skeleton disappear/flicker.
    center_x = center_x.fillna(center_x.median())
    center_y = center_y.fillna(center_y.median())
    center_z = center_z.fillna(center_z.median())

    for col in df.columns:
        if col.endswith('_x'):
            df[col] -= center_x
        elif col.endswith('_y'):
            df[col] -= center_y
        elif col.endswith('_z'):
            df[col] -= center_z

    return df


def process_dataset(dataset_id):
    """Center a single dataset."""
    print(f"\n  Centering {dataset_id}...")

    input_files = get_stage_files('Unified', dataset_id)
    output_dir = get_output_dir('Centered', dataset_id)

    if not input_files:
        print(f"    No files found for {dataset_id}")
        return

    for path in tqdm(input_files, desc=f"  {dataset_id}"):
        filename = os.path.basename(path)
        df = load_csv(path)

        # Check LEnd/REnd exist
        required = ['LEnd_x', 'LEnd_y', 'LEnd_z', 'REnd_x', 'REnd_y', 'REnd_z']
        if not all(col in df.columns for col in required):
            print(f"    WARNING: {filename} missing LEnd/REnd columns, copying as-is")
            save_csv(df, os.path.join(output_dir, filename))
            continue

        df = center_to_end_midpoint(df)
        save_csv(df, os.path.join(output_dir, filename))

    print(f"  Done: {len(input_files)} files")


def run():
    """Run Step 3 for all datasets."""
    print("\n" + "=" * 60)
    print("  STEP 3: Center Coordinates (LEnd-REnd midpoint → origin)")
    print("=" * 60)

    for dataset_id in DATASET_CONFIGS:
        process_dataset(dataset_id)

    print("\n  Step 3 complete.")


if __name__ == '__main__':
    run()
