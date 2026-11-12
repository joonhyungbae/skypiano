#!/bin/bash
#=============================================================================
# Stage 09: Visual3D body kinematics (segment endpoints + segment centers)
#=============================================================================
# Targets: per-frame 13-segment endpoint TSV/CSV (`*_body_segments.*`) and
# per-segment center TSV (`*_body_segment_centers.*`) from Visual3D-style
# marker data. Used by the explorer/viewer for stick-figure rendering.
#
# Entry scripts in steps/ (each takes its own --in / --out / --root flags).
# s01-s03 are alternative inputs that each write segments; s04 and s05 read
# their output or the per-trial motion CSVs:
#
#   s01_segment_body_csv.py         Explorer-format body marker CSV (24-year
#                                   piano sessions, e.g. 240828) -> segments.
#                                   Canonical entry for the public release.
#   s02_segment_piano_raw.py        Qualisys piano-style raw TSV -> segments.
#   s03_segment_qtm_raw.py          Qualisys QTM full-body raw TSV -> segments.
#   s04_compute_segment_centers.py  Derive segment midpoints from existing
#                                   *_body_segments.tsv files.
#   s05_compute_kinematics.py       Pseudo-kinematics (trunk lean, fingertip
#                                   velocity, hand span) per trial directory.
#
# Behavior:
#   --help / -h    print usage and exit
#   <args>         forward args to the canonical body_csv path
#   (no args)      print usage and exit (Stage 09 needs an explicit input)
#=============================================================================
set -euo pipefail
cd "$(dirname "$0")"

usage() {
    cat <<'USAGE'
Stage 09 - Visual3D body kinematics

Usage:
  run_stage.sh --in <body.csv> [--out <body_segments.csv>] [--no-centers]
      Run the canonical steps/s01_segment_body_csv.py path.

Alternative entry scripts (invoke directly):
  python steps/s01_segment_body_csv.py         --in <stem>_body.csv --out <stem>_body_segments.csv
  python steps/s02_segment_piano_raw.py        --in <raw>.tsv      --out <raw>_body_segments.tsv
  python steps/s03_segment_qtm_raw.py          --root <dir>
  python steps/s04_compute_segment_centers.py  --root <dir>
  python steps/s05_compute_kinematics.py       <trial_dir> [<trial_dir> ...]

Stage 09 has multiple upstream marker conventions, so the caller must pick
which entry script applies. The default forward path is the Explorer-CSV
flow used by the public 240828 release.
USAGE
}

if [[ $# -eq 0 || "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
    usage
    exit 0
fi

# Forward all args to the canonical Explorer-CSV entry point.
exec python steps/s01_segment_body_csv.py "$@"
