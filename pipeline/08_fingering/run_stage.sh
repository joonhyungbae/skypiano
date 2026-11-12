#!/bin/bash
#=============================================================================
# Stage 08: Per-note fingering — pipeline runner
#=============================================================================
# Steps (logical order == numeric order). They read and write Data/ relative
# to the repository root, so this runner runs them from there.
#
#   s01 pseudo_fingering     Tier A labels per piece -> Fingering/{dataset}/
#   s02 export_annotations   annotate-tool verdicts -> human_annotations.json
#   s03 train_tier_c         train the Tier C BiLSTM (opt-in, see below)
#   s04 release_fingering    fingering per released performance (Tiers A/B/C)
#
# Usage:
#   ./run_stage.sh --version X.Y.Z --hand raw          # s01, s02, s04
#   ./run_stage.sh --version X.Y.Z --hand imputed --dataset 240828
#   STEPS="s04" ./run_stage.sh --version X.Y.Z --hand raw --overwrite
#   STEPS="s01 s03" TRAIN_ARGS="--data mix --overwrite" ./run_stage.sh
#
# Arguments are forwarded to s04 only (s01 and s02 take none); s04 needs
# --version and --hand. Training is not in the default STEPS: the released
# Tier C weights already exist, so s03 runs only when STEPS names it.
#
# Environment variables:
#   STEPS        steps to run (default: s01 s02 s04)
#   TRAIN_ARGS   arguments for s03 (default: --data mix)
#=============================================================================
set -euo pipefail
cd "$(dirname "$0")/../.."

STEPS="${STEPS:-s01 s02 s04}"
TRAIN_ARGS="${TRAIN_ARGS:---data mix}"
STEP_DIR="pipeline/08_fingering/steps"

if [[ " $STEPS " == *" s04 "* && $# -eq 0 ]]; then
  echo "s04 needs --version X.Y.Z --hand raw|imputed (see the header of $0)"
  exit 1
fi

echo "================================================================"
echo "Stage 08: Per-note fingering"
echo "================================================================"
echo "Steps: $STEPS"
echo "Args:  $*"

run_step () {
  local name="$1"; shift
  echo ""
  echo "---- $name ----"
  python -u "$@"
}

for step in $STEPS; do
  case "$step" in
    s01) run_step "s01 pseudo fingering"    "$STEP_DIR/s01_pseudo_fingering.py" ;;
    s02) run_step "s02 export annotations"  "$STEP_DIR/s02_export_annotations.py" ;;
    # shellcheck disable=SC2086
    s03) run_step "s03 train Tier C"        "$STEP_DIR/s03_train_tier_c.py" $TRAIN_ARGS ;;
    s04) run_step "s04 release fingering"   "$STEP_DIR/s04_release_fingering.py" "$@" ;;
    *) echo "Unknown step: $step"; exit 1 ;;
  esac
done

echo ""
echo "================================================================"
echo "Stage 08 complete. Output: Data/Pipeline/08_fingering/, Data/web_annotate/data/"
echo "================================================================"
