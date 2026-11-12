"""
Marker definitions and column mappings for each dataset.

Unified naming convention:
  - Joints:  {L|R}{Joint}_x/y/z  (e.g., LWrist_x, RElbow_z)
  - End markers: LEnd, REnd (primary, all datasets)
  - Aux end markers: LEnd1, REnd1 (4-point datasets only, for leveling)

`missing_files` and `missing_restore` per dataset are sourced from the
`marker_restored` entries in `pipeline/_common/known_absences.json` (single
source of truth). Add new restorations there, not in this dict.
"""

import sys
from pathlib import Path as _Path

sys.path.insert(0, str(_Path(__file__).resolve().parents[3]))
from _common.known_absences import restored_marker_config as _restored_marker_config

# ============================================================
# Canonical joint set (matches imp/src/utils.py joint_cols)
# These 26 joints × 3 axes = 78 columns are what downstream expects
# ============================================================

CANONICAL_JOINTS = [
    # Left hand — 4 joints per finger: M (MCP) → P (PIP) → D (DIP) → F (fingertip)
    # Thumb: TC (CMC) → TM (MCP) → TD (IP) → TF (tip)
    'LIM', 'LIP', 'LID', 'LIF',
    'LMM', 'LMP', 'LMD', 'LMF',
    'LRM', 'LRP', 'LRD', 'LRF',
    'LLM', 'LLP', 'LLD', 'LLF',
    'LTC', 'LTM', 'LTD', 'LTF',
    'LElbow', 'LSh', 'LWrist',
    # Right hand
    'RIM', 'RIP', 'RID', 'RIF',
    'RMM', 'RMP', 'RMD', 'RMF',
    'RRM', 'RRP', 'RRD', 'RRF',
    'RLM', 'RLP', 'RLD', 'RLF',
    'RTC', 'RTM', 'RTD', 'RTF',
    'RElbow', 'RSh', 'RWrist',
]

# End markers: kept for alignment, not for training
END_MARKERS = ['LEnd', 'REnd']
END_MARKERS_AUX = ['LEnd1', 'REnd1']  # 4-point leveling (240809, 240828, 240523)
END_MARKERS_4PT = END_MARKERS + END_MARKERS_AUX

# All columns that Stage 02 hand outputs
OUTPUT_JOINTS = CANONICAL_JOINTS + END_MARKERS + END_MARKERS_AUX

# Build the full column list (joint_x, joint_y, joint_z for each)
CANONICAL_COLUMNS = [f'{j}_{a}' for j in CANONICAL_JOINTS for a in 'xyz']
END_COLUMNS = [f'{j}_{a}' for j in END_MARKERS for a in 'xyz']
END_AUX_COLUMNS = [f'{j}_{a}' for j in END_MARKERS_AUX for a in 'xyz']
OUTPUT_COLUMNS = CANONICAL_COLUMNS + END_COLUMNS + END_AUX_COLUMNS


# ============================================================
# Dataset configurations
# ============================================================

DATASET_CONFIGS = {
    '240809': {
        # Migrated to Stage 01 canonical output (was: 240809/C0/N0/Motion/Hand).
        # File stems are now canonical like 240809_C0_N0_T2_P3_t1.csv.
        'raw_path': 'Pipeline/01_ingest/Normalized/240809/C0_N0/motion_hand',
        'raw_glob': '*.csv',
        # Keep TimeCode — Stage 03 midi_motion s03 needs it to re-extract session MIDI
        # aligned to Stage 02's possibly-trimmed frame range.
        'columns_to_drop': ['Time'],
        # Stage 01 copies the raw Motive export (metadata header + Name row).
        'needs_dot_conversion': False,
        'has_metadata_header': True,
        # Motive X/Y/Z land on _z/_x/_y: the frame the pre-Stage-01 inputs
        # used, which reference_params.npz (every dataset's target) is built in.
        'axis_order': ('z', 'x', 'y'),

        # Lend2 → LEnd (primary), Lend1 → LEnd1 (aux)
        'column_mapping': {
            'Lend2_x': 'LEnd_x', 'Lend2_y': 'LEnd_y', 'Lend2_z': 'LEnd_z',
            'Rend2_x': 'REnd_x', 'Rend2_y': 'REnd_y', 'Rend2_z': 'REnd_z',
            'Lend1_x': 'LEnd1_x', 'Lend1_y': 'LEnd1_y', 'Lend1_z': 'LEnd1_z',
            'Rend1_x': 'REnd1_x', 'Rend1_y': 'REnd1_y', 'Rend1_z': 'REnd1_z',
            'Lelbow_x': 'LElbow_x', 'Lelbow_y': 'LElbow_y', 'Lelbow_z': 'LElbow_z',
            'Relbow_x': 'RElbow_x', 'Relbow_y': 'RElbow_y', 'Relbow_z': 'RElbow_z',
            'Lwrist_x': 'LWrist_x', 'Lwrist_y': 'LWrist_y', 'Lwrist_z': 'LWrist_z',
            'Rwrist_x': 'RWrist_x', 'Rwrist_y': 'RWrist_y', 'Rwrist_z': 'RWrist_z',
            'Lsh_x': 'LSh_x', 'Lsh_y': 'LSh_y', 'Lsh_z': 'LSh_z',
            'Rsh_x': 'RSh_x', 'Rsh_y': 'RSh_y', 'Rsh_z': 'RSh_z',
        },

        # Session-specific override (now canonical P5_t1): in this session's raw
        # export, the "Rend1" channel is mislabeled and corresponds to Lend2.
        # Remap Rend1→Lend2 before standard mapping.
        'file_column_overrides': {
            '240809_C0_N0_T2_P5_t1.csv': {
                'Rend1_x': 'Lend2_x', 'Rend1_y': 'Lend2_y', 'Rend1_z': 'Lend2_z',
            },
        },


        'fps': 120,
        'fps_conversion': False,
    },

    '211215': {
        # Migrated to Stage 01 canonical (was: 211215/C0/N0/Motion/Hand).
        # Stems carry canonical T9 (raw used T3 incorrectly).
        'raw_path': 'Pipeline/01_ingest/Normalized/211215/C0_N0/motion',
        'raw_glob': '*.csv',
        'columns_to_drop': ['Unnamed: 0.1', 'Unnamed: 0'],
        'needs_dot_conversion': False,
        'has_metadata_header': False,

        'column_mapping': {
            'L-END_x': 'LEnd_x', 'L-END_y': 'LEnd_y', 'L-END_z': 'LEnd_z',
            'R-END_x': 'REnd_x', 'R-END_y': 'REnd_y', 'R-END_z': 'REnd_z',
            'LLAE_x': 'LElbow_x', 'LLAE_y': 'LElbow_y', 'LLAE_z': 'LElbow_z',
            'RLAE_x': 'RElbow_x', 'RLAE_y': 'RElbow_y', 'RLAE_z': 'RElbow_z',
            'LMDW_x': 'LWrist_x', 'LMDW_y': 'LWrist_y', 'LMDW_z': 'LWrist_z',
            'RMDW_x': 'RWrist_x', 'RMDW_y': 'RWrist_y', 'RMDW_z': 'RWrist_z',
            'LSH_x': 'LSh_x', 'LSH_y': 'LSh_y', 'LSH_z': 'LSh_z',
            'RSH_x': 'RSh_x', 'RSH_y': 'RSh_y', 'RSH_z': 'RSh_z',
        },

        'fps': 120,          # Vicon export at 120 Hz (100 stretched time 1.2x)
        'fps_conversion': False,
    },

    '220819': {
        # Migrated to Stage 01 canonical (was: Raw/220819/{C}/{N}/Motion/Hand).
        # Stage 01 merges the 5 sessions into canonical C#_N# session dirs.
        'raw_path': [
            'Pipeline/01_ingest/Normalized/220819/C0_N0/motion_hand',
            'Pipeline/01_ingest/Normalized/220819/C0_N1/motion_hand',
            'Pipeline/01_ingest/Normalized/220819/C0_N2/motion_hand',
            'Pipeline/01_ingest/Normalized/220819/C1_N0/motion_hand',
            'Pipeline/01_ingest/Normalized/220819/C1_N1/motion_hand',
        ],
        'raw_glob': '*.csv',
        'columns_to_drop': [],
        'needs_dot_conversion': False,
        'has_metadata_header': 'auto',  # sliced=False, Raw=True

        # Wrist = midpoint of the radial (LAW) and ulnar (MEW) styloid markers,
        # i.e. the wrist joint centre. A single styloid marker sits ~35 mm off
        # the joint axis on either side (see `wrist_center` in the unify step).
        'wrist_center': {
            'LWrist': ('LLAW', 'LMEW'),
            'RWrist': ('RLAW', 'RMEW'),
        },
        'column_mapping': {
            'LEND_x': 'LEnd_x', 'LEND_y': 'LEnd_y', 'LEND_z': 'LEnd_z',
            'REND_x': 'REnd_x', 'REND_y': 'REnd_y', 'REND_z': 'REnd_z',
            'LLAE_x': 'LElbow_x', 'LLAE_y': 'LElbow_y', 'LLAE_z': 'LElbow_z',
            'RLAE_x': 'RElbow_x', 'RLAE_y': 'RElbow_y', 'RLAE_z': 'RElbow_z',
            'LSH_x': 'LSh_x', 'LSH_y': 'LSh_y', 'LSH_z': 'LSh_z',
            'RSH_x': 'RSh_x', 'RSH_y': 'RSh_y', 'RSH_z': 'RSh_z',
        },

        # missing_files / missing_restore sourced from known_absences registry.
        'fps': 100,
        'fps_conversion': True,
    },

    # ── Metadata header datasets ──────────────────────────────

    '240828': {
        # Migrated to Stage 01 canonical (Task 0/1/2/9 split per docx labels).
        'raw_path': [
            'Pipeline/01_ingest/Normalized/240828/C0_N4/motion_hand',
            'Pipeline/01_ingest/Normalized/240828/C0_N5/motion_hand',
            'Pipeline/01_ingest/Normalized/240828/C0_N6/motion_hand',
            'Pipeline/01_ingest/Normalized/240828/C0_N7/motion_hand',
            'Pipeline/01_ingest/Normalized/240828/C0_N8/motion_hand',
        ],
        'raw_glob': '*.csv',
        'columns_to_drop': [],
        'needs_dot_conversion': False,
        'has_metadata_header': True,

        'column_mapping': {
            'Lend2_x': 'LEnd_x', 'Lend2_y': 'LEnd_y', 'Lend2_z': 'LEnd_z',
            'Rend2_x': 'REnd_x', 'Rend2_y': 'REnd_y', 'Rend2_z': 'REnd_z',
            'Lend1_x': 'LEnd1_x', 'Lend1_y': 'LEnd1_y', 'Lend1_z': 'LEnd1_z',
            'Rend1_x': 'REnd1_x', 'Rend1_y': 'REnd1_y', 'Rend1_z': 'REnd1_z',
            'Lelbow_x': 'LElbow_x', 'Lelbow_y': 'LElbow_y', 'Lelbow_z': 'LElbow_z',
            'Relbow_x': 'RElbow_x', 'Relbow_y': 'RElbow_y', 'Relbow_z': 'RElbow_z',
            'Lwrist_x': 'LWrist_x', 'Lwrist_y': 'LWrist_y', 'Lwrist_z': 'LWrist_z',
            'Rwrist_x': 'RWrist_x', 'Rwrist_y': 'RWrist_y', 'Rwrist_z': 'RWrist_z',
            'Lsh_x': 'LSh_x', 'Lsh_y': 'LSh_y', 'Lsh_z': 'LSh_z',
            'Rsh_x': 'RSh_x', 'Rsh_y': 'RSh_y', 'Rsh_z': 'RSh_z',
        },

        # 4 trials have Lend2↔Rend1 labels swapped in raw mocap output.
        # Detected geometrically: Lend1-Lend2 distance ~1270mm (keyboard span)
        # instead of normal ~81mm (L-side aux offset). Swapping restores the
        # correct L/R pairing before standard column_mapping applies.
        'file_column_overrides': {
            '240828_C0_N4_T2_P6_t1.csv': {
                'Lend2_x': 'Rend1_x', 'Lend2_y': 'Rend1_y', 'Lend2_z': 'Rend1_z',
                'Rend1_x': 'Lend2_x', 'Rend1_y': 'Lend2_y', 'Rend1_z': 'Lend2_z',
            },
            '240828_C0_N5_T1_P4_t1.csv': {
                'Lend2_x': 'Rend1_x', 'Lend2_y': 'Rend1_y', 'Lend2_z': 'Rend1_z',
                'Rend1_x': 'Lend2_x', 'Rend1_y': 'Lend2_y', 'Rend1_z': 'Lend2_z',
            },
            '240828_C0_N5_T1_P12_t1.csv': {
                'Lend2_x': 'Rend1_x', 'Lend2_y': 'Rend1_y', 'Lend2_z': 'Rend1_z',
                'Rend1_x': 'Lend2_x', 'Rend1_y': 'Lend2_y', 'Rend1_z': 'Lend2_z',
            },
            '240828_C0_N8_T1_P6_t1.csv': {
                'Lend2_x': 'Rend1_x', 'Lend2_y': 'Rend1_y', 'Lend2_z': 'Rend1_z',
                'Rend1_x': 'Lend2_x', 'Rend1_y': 'Lend2_y', 'Rend1_z': 'Lend2_z',
            },
        },

        # missing_files / missing_restore sourced from known_absences registry.
        'fps': 120,
        'fps_conversion': False,
    },

    '221112': {
        # Migrated to Stage 01 canonical (one session dir per subject N2..N11).
        'raw_path': [
            f'Pipeline/01_ingest/Normalized/221112/C1_N{n}/motion_hand' for n in range(2, 12)
        ],
        'raw_glob': '*.csv',
        'columns_to_drop': [],
        'needs_dot_conversion': False,
        'has_metadata_header': True,

        # Wrist = midpoint of the radial (LAW) and ulnar (MEW) styloid markers,
        # i.e. the wrist joint centre. A single styloid marker sits ~35 mm off
        # the joint axis on either side (see `wrist_center` in the unify step).
        'wrist_center': {
            'LWrist': ('LLAW', 'LMEW'),
            'RWrist': ('RLAW', 'RMEW'),
        },
        'column_mapping': {
            'LEND_x': 'LEnd_x', 'LEND_y': 'LEnd_y', 'LEND_z': 'LEnd_z',
            'REND_x': 'REnd_x', 'REND_y': 'REnd_y', 'REND_z': 'REnd_z',
            'LLAE_x': 'LElbow_x', 'LLAE_y': 'LElbow_y', 'LLAE_z': 'LElbow_z',
            'RLAE_x': 'RElbow_x', 'RLAE_y': 'RElbow_y', 'RLAE_z': 'RElbow_z',
            'LSH_x': 'LSh_x', 'LSH_y': 'LSh_y', 'LSH_z': 'LSh_z',
            'RSH_x': 'RSh_x', 'RSH_y': 'RSh_y', 'RSH_z': 'RSh_z',
        },

        'fps': 100,
        'fps_conversion': True,
    },

    '240523': {
        # Migrated to Stage 01 canonical (Task 9 free-choice pieces).
        'raw_path': 'Pipeline/01_ingest/Normalized/240523/C1_N0/motion_hand',
        'raw_glob': '*.csv',
        'columns_to_drop': [],
        'needs_dot_conversion': False,
        'has_metadata_header': True,

        # 240523 has 1/2 numbering swapped vs 240809:
        # Lend1/Rend1 are the keyboard-edge markers (→ primary LEnd/REnd)
        # Lend2/Rend2 are the offset markers (→ aux LEnd1/REnd1)
        'column_mapping': {
            'Lend1_x': 'LEnd_x', 'Lend1_y': 'LEnd_y', 'Lend1_z': 'LEnd_z',
            'Rend1_x': 'REnd_x', 'Rend1_y': 'REnd_y', 'Rend1_z': 'REnd_z',
            'Lend2_x': 'LEnd1_x', 'Lend2_y': 'LEnd1_y', 'Lend2_z': 'LEnd1_z',
            'Rend2_x': 'REnd1_x', 'Rend2_y': 'REnd1_y', 'Rend2_z': 'REnd1_z',
            'Lelbow_x': 'LElbow_x', 'Lelbow_y': 'LElbow_y', 'Lelbow_z': 'LElbow_z',
            'Relbow_x': 'RElbow_x', 'Relbow_y': 'RElbow_y', 'Relbow_z': 'RElbow_z',
            'Lwrist_x': 'LWrist_x', 'Lwrist_y': 'LWrist_y', 'Lwrist_z': 'LWrist_z',
            'Rwrist_x': 'RWrist_x', 'Rwrist_y': 'RWrist_y', 'Rwrist_z': 'RWrist_z',
            'Lsh_x': 'LSh_x', 'Lsh_y': 'LSh_y', 'Lsh_z': 'LSh_z',
            'Rsh_x': 'RSh_x', 'Rsh_y': 'RSh_y', 'Rsh_z': 'RSh_z',
        },

        'fps': 120,
        'fps_conversion': False,
    },
}

REFERENCE_DATE = '240809'
TARGET_DATES = ['211215', '220819', '221112', '240523', '240828']


# Inject missing_files / missing_restore from the known_absences registry so
# that this module remains the sole DATASET_CONFIGS surface but the underlying
# data lives in pipeline/_common/known_absences.json.
for _ds, _cfg in DATASET_CONFIGS.items():
    _files, _restore = _restored_marker_config(_ds)
    _cfg['missing_files'] = _files
    _cfg['missing_restore'] = _restore


def get_coordinate_columns(df):
    """Get all coordinate column names from a DataFrame."""
    return [col for col in df.columns if col.endswith(('_x', '_y', '_z'))]


def get_marker_names(df):
    """Extract unique marker names from coordinate columns."""
    markers = set()
    for col in df.columns:
        for suffix in ('_x', '_y', '_z'):
            if col.endswith(suffix):
                markers.add(col[:-len(suffix)])
                break
    return sorted(markers)


def filter_to_stage2_columns(df):
    """
    Filter DataFrame to only keep OUTPUT_COLUMNS (plus TimeCode if present).
    Missing columns are added as NaN. Extra columns are dropped.

    TimeCode is preserved when the raw recording has it — downstream stages
    (Stage 03 midi_motion s03) use it to re-extract session MIDI aligned to Stage 02's
    possibly-trimmed frame range.
    """
    for col in OUTPUT_COLUMNS:
        if col not in df.columns:
            df[col] = float('nan')
    keep = list(OUTPUT_COLUMNS)
    if 'TimeCode' in df.columns:
        keep = ['TimeCode'] + keep
    return df[keep]


def apply_wrist_center(df, config):
    """
    Add {side}Wrist_{x,y,z} as the per-frame midpoint of two raw markers for
    datasets whose config defines `wrist_center` (styloid pair -> joint centre).
    Frames where either source marker is missing stay NaN.
    """
    for joint, (a, b) in config.get('wrist_center', {}).items():
        for ax in ('x', 'y', 'z'):
            ca, cb = f'{a}_{ax}', f'{b}_{ax}'
            if ca in df.columns and cb in df.columns:
                df[f'{joint}_{ax}'] = (df[ca] + df[cb]) / 2.0
    return df
