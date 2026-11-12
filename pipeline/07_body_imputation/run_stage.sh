#!/bin/bash
#=============================================================================
# Stage 07: Full-Body Motion Imputation — pipeline runner
#=============================================================================
# Usage:
#   ./run_stage.sh                      # run all steps
#   ./run_stage.sh --skip C0_N0 C1_N0
#   ./run_stage.sh --only C0_N4
#   STEPS="s02" ./run_stage.sh --only C0_N4
#
# Environment:
#   STEPS    steps to run (default: s01 s02 s03 s04)
#=============================================================================
set -euo pipefail
cd "$(dirname "$0")"

ARGS="$@"
STEPS="${STEPS:-s01 s02 s03 s04}"

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
    s01) run_module "s01 slice body data"   steps.s01_slice_data ;;
    s02) run_module "s02 SAITS train"       steps.s02_saits_train ;;
    s03) run_module "s03 SAITS imputation"  steps.s03_impute ;;
    s04) run_module "s04 replace NaN"       steps.s04_replacement ;;
    *) echo "Unknown step: $step"; exit 1 ;;
  esac
done

echo ""
echo "================================================================"
echo "Stage 07 complete. Final output: Data/Pipeline/07_body_imputation/Final/{pid}/"
echo "================================================================"
