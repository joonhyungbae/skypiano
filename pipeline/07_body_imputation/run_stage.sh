#!/bin/bash
#=============================================================================
# Stage 07: body outlier removal and gap filling — pipeline runner
#=============================================================================
# Usage:
#   ./run_stage.sh                      # run all steps
#   ./run_stage.sh --skip C0_N0 C1_N0
#   ./run_stage.sh --only C0_N4
#   STEPS="s02" ./run_stage.sh --only C0_N4
#
# Environment:
#   STEPS    steps to run (default: s00 s04; s01-s03 = the earlier SAITS model, s05/s06 = held-out accuracy, on request)
#=============================================================================
set -euo pipefail
# Data paths are relative to the repository root; the step modules live in this folder.
HERE="$(cd "$(dirname "$0")" && pwd)"
cd "$HERE/../.."
export PYTHONPATH="$(cygpath -w "$HERE" 2>/dev/null || echo "$HERE")"   # Windows Python needs a native path

ARGS="$@"
STEPS="${STEPS:-s00 s04}"

echo "================================================================"
echo "Stage 07: Full-Body Motion Imputation"
echo "================================================================"
echo "Steps: $STEPS"
echo "Args:  $ARGS"

run_module () {
  local name="$1"; shift
  echo ""
  echo "---- $name ----"
  python -u -m "$@" $ARGS
}

for step in $STEPS; do
  case "$step" in
    s00) run_module "s00 outlier removal"   steps.s00_outlier_detect ;;
    s01) run_module "s01 slice body data"   steps.s01_slice_data ;;
    s02) run_module "s02 SAITS train"       steps.s02_saits_train ;;
    s03) run_module "s03 SAITS imputation"  steps.s03_impute ;;
    s04) run_module "s04 fill gaps (rule)"  steps.s04_replacement ;;
    s06) run_module "s06 compare fills"     steps.s06_compare_fill ;;
    s05) run_module "s05 held-out accuracy" steps.s05_evaluate ;;
    *) echo "Unknown step: $step"; exit 1 ;;
  esac
done

echo ""
echo "================================================================"
echo "Stage 07 complete. Final output: Data/Pipeline/07_body_imputation/Final/{pid}/"
echo "================================================================"
