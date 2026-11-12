"""
Step 2: Restore Missing End Markers

- 240809 (canonical P5_t1): missing LEnd (=Lend2) → predict from LEnd1 (=Lend1)
- 220819 T3_t2: missing LEnd → predict from REnd
- Uses LinearRegression trained on files with complete data

Input:  Data/Pipeline/02_spatial/hand/Unified/{dataset_id}/*.csv
Output: Data/Pipeline/02_spatial/hand/Unified/{dataset_id}/*.csv (in-place update)
"""

import pandas as pd
import numpy as np
import os
from tqdm import tqdm

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.markers import DATASET_CONFIGS
from src.io_utils import get_stage_files, load_csv, save_csv


def train_restoration_model(file_paths, target_cols, input_cols):
    """
    Build an OFFSET-BASED model: target = input + median_offset.

    End markers are rigidly attached to the piano, so the spatial offset
    between target and input markers should be constant across files (same
    piano, same setup). Median offset is more robust than linear regression
    when the relationship is a fixed translation.

    Args:
        file_paths: list of CSV paths with complete data
        target_cols: columns to predict (e.g., ['LEnd_x', 'LEnd_y', 'LEnd_z'])
        input_cols: columns to use as features (e.g., ['LEnd1_x', 'LEnd1_y', 'LEnd1_z'])

    Returns:
        OffsetModel with .predict(X) method
    """
    all_offsets = []

    for path in file_paths:
        df = load_csv(path)
        required = target_cols + input_cols
        if not all(col in df.columns for col in required):
            continue

        valid = df[required].dropna()
        if len(valid) > 0:
            # Per-file median offset (target - input) for each axis
            tgt = valid[target_cols].median().values
            inp = valid[input_cols].median().values
            if len(tgt) == len(inp):
                all_offsets.append(tgt - inp)

    if not all_offsets:
        raise ValueError(f"No training data found. Targets: {target_cols}, Inputs: {input_cols}")

    offset = np.median(all_offsets, axis=0)
    print(f"    Fitted offset from {len(all_offsets)} files: {offset.round(1)}")

    class OffsetModel:
        def __init__(self, offset):
            self.offset = offset
        def predict(self, X):
            return X + self.offset

    return OffsetModel(offset)


def restore_missing_in_file(df, model, target_cols, input_cols):
    """Apply trained offset model, but ONLY fill columns that are entirely missing.

    We don't overwrite existing target values — restoration is strictly
    additive: it fills columns whose values are all NaN across the file.
    """
    # Ensure target columns exist
    for col in target_cols:
        if col not in df.columns:
            df[col] = np.nan

    # Identify which target columns are entirely missing (those are the ones
    # we need to fill). A mix of present + NaN in a target column means
    # the marker was tracked partially — we leave it alone.
    cols_to_restore = [c for c in target_cols if df[c].isna().all()]
    if not cols_to_restore:
        print(f"    Nothing to restore (all target columns populated)")
        return df

    # For offset model, we need the full vector prediction, then copy only
    # the columns we want to restore.
    if any(c not in df.columns for c in input_cols):
        print(f"    Missing input columns, skipping")
        return df

    input_data = df[input_cols]
    valid_mask = ~input_data.isna().any(axis=1)
    if valid_mask.sum() == 0:
        print(f"    No valid input rows, skipping")
        return df

    X = input_data[valid_mask].values
    predictions = model.predict(X)  # full 6-vector or N-vector

    # Map target col → prediction column index
    for i, tcol in enumerate(target_cols):
        if tcol in cols_to_restore:
            df.loc[valid_mask[valid_mask].index, tcol] = predictions[:, i]

    print(f"    Restored columns: {cols_to_restore} ({valid_mask.sum()} frames)")
    return df


def process_dataset(dataset_id, config):
    """Restore missing markers for a single dataset."""
    if not config['missing_files'] or config.get('missing_restore') is None:
        return

    print(f"\n  {dataset_id}: {len(config['missing_files'])} files need restoration")

    restore_cfg = config['missing_restore']
    target_cols = restore_cfg['target_cols']
    input_cols = restore_cfg['input_cols']

    all_files = get_stage_files('Unified', dataset_id)
    missing_filenames = set(config['missing_files'])

    # Split into training files and files needing restoration
    training_files = [f for f in all_files if os.path.basename(f) not in missing_filenames]
    missing_files = [f for f in all_files if os.path.basename(f) in missing_filenames]

    if not missing_files:
        print(f"    Missing files not found in Unified output, skipping")
        return

    # Train model
    print(f"    Training restoration model...")
    print(f"    Target: {target_cols}")
    print(f"    Input:  {input_cols}")
    model = train_restoration_model(training_files, target_cols, input_cols)

    # Apply restoration
    for path in missing_files:
        filename = os.path.basename(path)
        print(f"    Restoring {filename}...")
        df = load_csv(path)
        df = restore_missing_in_file(df, model, target_cols, input_cols)
        save_csv(df, path)


def _session_map(dataset_id):
    """File stem -> session (C#_N#), for canonical and raw stems (Stage 01 mappings)."""
    import json
    import re
    out = {}
    root = os.path.join('Data', 'Pipeline', '01_ingest', 'Normalized', dataset_id)
    if os.path.isdir(root):
        for sess in os.listdir(root):
            mp = os.path.join(root, sess, '_mapping.json')
            if not os.path.isfile(mp):
                continue
            with open(mp, encoding='utf-8') as f:
                for t in json.load(f).get('trials', []):
                    out[t['trial_stem']] = sess
                    if t.get('raw_stem'):
                        out[t['raw_stem']] = sess
    def session_of(path):
        stem = os.path.splitext(os.path.basename(path))[0]
        if stem in out:
            return out[stem]
        m = re.search(r'C\d+_N\d+', stem)
        return m.group(0) if m else None
    return session_of


AUTO_RESTORE_EXCLUDE = {
    # 240809 is the cross-year reference dataset. Modifying its Unified data
    # cascades into s04_level_240809 -> reference_params.npz -> every other
    # dataset's s05 alignment. Leave it manually managed.
    '240809',
}


def auto_restore_ends(dataset_id):
    """
    Auto-detect all-NaN End marker columns and restore via offset regression.

    Piano-edge End markers (LEnd/REnd/LEnd1/REnd1) are rigidly attached to a
    stationary piano, so the spatial offset between any pair is a constant
    translation per capture setup. The setup differs between sessions: on
    240828 the LEnd-REnd vector keeps its length (1267 mm) but its direction
    in capture coordinates differs by up to ~60 mm between sessions, while it
    varies < 1-2 mm between the trials of one session. So the offset is
    learned from trials of the SAME session where both markers are valid (the
    dataset-wide median is used only if the session has none, with a
    warning), and the missing column is filled from the partner marker.

    Runs BEFORE config-driven `process_dataset` so pattern fixes (missing_files)
    can remain for non-End edge cases. Idempotent: silently no-ops when
    nothing needs restoration.
    """
    if dataset_id in AUTO_RESTORE_EXCLUDE:
        return
    # Target → preferred input chain. If primary input is also NaN for this
    # file, try fallback.
    END_CHAINS = [
        ('LEnd',  ['REnd',  'LEnd1']),
        ('REnd',  ['LEnd',  'REnd1']),
        ('LEnd1', ['REnd1', 'LEnd']),
        ('REnd1', ['LEnd1', 'REnd']),
    ]

    all_files = get_stage_files('Unified', dataset_id)
    if not all_files:
        return

    # Pass 1: per-file, decide which target to restore and from which input
    jobs = []  # (path, target, input)
    for path in all_files:
        df = load_csv(path)
        for tgt, inputs in END_CHAINS:
            tgt_col = f'{tgt}_x'
            if tgt_col not in df.columns:
                continue
            if not df[tgt_col].isna().all():
                continue  # not a silent-fail case
            # pick first available input
            for cand in inputs:
                cand_col = f'{cand}_x'
                if cand_col in df.columns and df[cand_col].notna().any():
                    jobs.append((path, tgt, cand))
                    break

    if not jobs:
        return

    print(f"\n  [{dataset_id}] auto-detect End markers: {len(jobs)} restoration candidates")

    # Pass 2: learn the offset per (target, input, session) from trials of the
    # same session where both markers are valid
    session_of = _session_map(dataset_id)
    jobs_paths = {j[0] for j in jobs}
    offsets = {}
    for pair in sorted(set((j[1], j[2]) for j in jobs)):
        tgt, inp = pair
        tgt_cols = [f'{tgt}_{a}' for a in ('x', 'y', 'z')]
        inp_cols = [f'{inp}_{a}' for a in ('x', 'y', 'z')]
        rows = {}   # session -> [offset]
        for path in all_files:
            if path in jobs_paths:
                continue
            df = load_csv(path)
            req = tgt_cols + inp_cols
            if not all(c in df.columns for c in req):
                continue
            valid = df[req].dropna()
            if len(valid) == 0:
                continue
            rows.setdefault(session_of(path), []).append(
                valid[tgt_cols].median().values - valid[inp_cols].median().values)
        if not rows:
            print(f"    {tgt} from {inp}: NO training samples - skipping (likely 2-point dataset)")
            continue
        all_rows = [r for v in rows.values() for r in v]
        for sess in {session_of(j[0]) for j in jobs if (j[1], j[2]) == pair}:
            if rows.get(sess):
                offsets[(tgt, inp, sess)] = np.median(rows[sess], axis=0)
                print(f"    [{sess}] {tgt} = {inp} + {offsets[(tgt, inp, sess)].round(1)}  ({len(rows[sess])} trials of the session)")
            else:
                offsets[(tgt, inp, sess)] = np.median(all_rows, axis=0)
                print(f"    WARNING [{sess}] {tgt} = {inp} + {offsets[(tgt, inp, sess)].round(1)}  "
                      f"(no trial of the session has both; dataset-wide median of {len(all_rows)})")

    # Pass 3: apply
    from collections import defaultdict
    by_file = defaultdict(list)
    for path, tgt, inp in jobs:
        if (tgt, inp, session_of(path)) in offsets:
            by_file[path].append((tgt, inp))
    for path, tgt_inp_list in by_file.items():
        df = load_csv(path)
        for tgt, inp in tgt_inp_list:
            tgt_cols = [f'{tgt}_{a}' for a in ('x', 'y', 'z')]
            inp_cols = [f'{inp}_{a}' for a in ('x', 'y', 'z')]
            mask = ~df[inp_cols].isna().any(axis=1)
            if mask.sum() == 0:
                continue
            pred = df.loc[mask, inp_cols].values + offsets[(tgt, inp, session_of(path))]
            for i, col in enumerate(tgt_cols):
                df.loc[mask, col] = pred[:, i]
            print(f"    [auto-restore] {os.path.basename(path)}: {tgt} <- {inp}  "
                  f"({mask.sum()}/{len(df)} frames = {mask.sum()/len(df)*100:.0f}%)")
        save_csv(df, path)


NEIGHBOUR_TOL_MM = 2.0


def fill_ends_from_neighbours(dataset_id):
    """
    A trial whose LEnd and REnd are both missing for the whole take has no
    partner to restore from. The piano does not move, so within one capture
    setup the end markers sit at the same capture coordinates in every trial
    (211215: within 0.5 mm, but the setup changed once, by ~120 mm). Such a
    trial takes the median end positions of its neighbours in recording order
    (the previous and the next trial of its session that have them), and only
    when those two agree within NEIGHBOUR_TOL_MM -- i.e. the trial lies inside
    one unchanged setup.
    """
    import re
    files = get_stage_files('Unified', dataset_id)
    if not files:
        return
    session_of = _session_map(dataset_id)
    cols = {m: [f'{m}_{a}' for a in 'xyz'] for m in ('LEnd', 'REnd')}

    def order(path):
        m = re.search(r'_P(\d+)_t(\d+)', os.path.basename(path))
        return (int(m.group(1)), int(m.group(2))) if m else (10 ** 6, 0)

    ends, missing = {}, []
    for path in files:
        df = load_csv(path)
        if not all(c in df.columns for v in cols.values() for c in v):
            continue
        if df[cols['LEnd'][0]].isna().all() and df[cols['REnd'][0]].isna().all():
            missing.append(path)
        else:
            med = {m: df[c].median().values for m, c in cols.items()}
            if all(np.isfinite(v).all() for v in med.values()):
                ends[path] = med
    for path in missing:
        same = sorted([q for q in ends if session_of(q) == session_of(path)], key=order)
        before = [q for q in same if order(q) < order(path)]
        after = [q for q in same if order(q) > order(path)]
        if not before or not after:
            print(f"    {os.path.basename(path)}: both ends missing, no neighbour on both sides - left as is")
            continue
        a, b = ends[before[-1]], ends[after[0]]
        gap = max(float(np.linalg.norm(a[m] - b[m])) for m in cols)
        if gap > NEIGHBOUR_TOL_MM:
            print(f"    {os.path.basename(path)}: neighbours disagree by {gap:.1f} mm - left as is")
            continue
        df = load_csv(path)
        for m, c in cols.items():
            v = (a[m] + b[m]) / 2
            for i, col in enumerate(c):
                df[col] = v[i]
        save_csv(df, path)
        print(f"    [neighbour-fill] {os.path.basename(path)}: LEnd/REnd from "
              f"{os.path.basename(before[-1])} and {os.path.basename(after[0])} (agree within {gap:.1f} mm)")


def run():
    """Run Step 2 for all datasets."""
    print("\n" + "=" * 60)
    print("  STEP 2: Restore Missing End Markers")
    print("=" * 60)

    # Phase A: auto-detect + restore End markers (LEnd/REnd/LEnd1/REnd1)
    # via piano-offset regression. Covers the silent-fail case where a marker
    # is entirely NaN in a capture. Runs independently of config.
    print("\n  Phase A: auto End-marker restoration")
    for dataset_id in DATASET_CONFIGS:
        auto_restore_ends(dataset_id)
        fill_ends_from_neighbours(dataset_id)

    # Phase B: config-driven restoration (missing_files + missing_restore).
    # Retained for cases that need explicit control (non-End markers, custom
    # input/target pairs, or when auto-detect should not run).
    print("\n  Phase B: config-driven restoration")
    for dataset_id, config in DATASET_CONFIGS.items():
        process_dataset(dataset_id, config)

    print("\n  Step 2 complete.")


if __name__ == '__main__':
    run()
