#!/bin/bash
#=============================================================================
# Stage 06: Hand Motion Imputation — full pipeline runner
#=============================================================================
# Runs all steps sequentially (Stage 04 per-piece input → Final output)
#
# Full pipeline (logical order == numeric order):
#   s01 slice_data           per-performer 3s window slicing
#   s02 saits_train          per-performer Body SAITS train (1st pass)
#   s03 impute_first_pass    per-performer 1st-pass imputation
#   s04 replacement          fill NaN with imputed values
#   s05 hand_split           split L/R + wrist alignment + rotation metadata
#   s06 outlier_detect       rule-based outlier → NaN (bone/vel/accel/pos)
#   s07 slice_hands          slice L/R hands (per-performer TrainData + non-overlapping inference)
#   s08 saits_train_hand     per-performer L/R SAITS train (one model per performer × hand side)
#   s09 impute_hands         2nd-pass SAITS imputation, per-performer model lookup
#   s10 merge                Imp/result → Imp/concat (file copy)
#   s11 restore              inverse rotation → restore original coordinates
#   s12 final_replacement    L+R → final merge into full body
#
# Usage:
#   ./run_stage.sh                      # run everything for all performers
#   ./run_stage.sh --skip C0_N0 C1_N0
#   ./run_stage.sh --only C0_N4
#   STEPS="s05" ./run_stage.sh --skip C0_N0 C1_N0    # run only specified steps
#
# Environment variables:
#   STEPS          steps to run (default: all)
#=============================================================================
set -euo pipefail
cd "$(dirname "$0")"

# Forward all args to steps that support --only/--skip
ARGS="$@"

# Default: all steps in order
STEPS="${STEPS:-s01 s02 s03 s04 s05 s06 s07 s08 s09 s10 s11 s12}"

echo "================================================================"
echo "Stage 06: Hand Motion Imputation Pipeline"
echo "================================================================"
echo "Steps: $STEPS"
echo "Args:  $ARGS"
echo ""

run_module () {
  local name="$1"; shift
  echo ""
  echo "---- $name ----"
  python -u -m "$@" $ARGS
}

for step in $STEPS; do
  case "$step" in
    s01) run_module "s01 slice data"                steps.s01_slice_data ;;
    s02) run_module "s02 Body SAITS train"          steps.s02_saits_train ;;
    s03) run_module "s03 1st-pass imputation"       steps.s03_impute_first_pass ;;
    s04) run_module "s04 replace NaN with imputed"  steps.s04_replacement ;;
    s05) run_module "s05 hand split + coord align"  steps.s05_hand_split ;;
    s06) run_module "s06 rule-based outlier detect" steps.s06_outlier_detect ;;
    s07) run_module "s07 slice L/R hands"           steps.s07_slice_hands ;;
    s08) run_module "s08 per-performer L/R hand SAITS train" steps.s08_saits_train_hand ;;
    s09) run_module "s09 2nd SAITS imputation"      steps.s09_impute_hands ;;
    s10) run_module "s10 merge sliced outputs"      steps.s10_merge ;;
    s11) run_module "s11 inverse coord restore"     steps.s11_restore ;;
    s12) run_module "s12 final L+R merge"           steps.s12_final_replacement ;;
    *) echo "Unknown step: $step"; exit 1 ;;
  esac
done

echo ""
echo "================================================================"
echo "Stage 06 complete. Final output: Data/Pipeline/06_hand_imputation/Final/{pid}/"
echo "================================================================"
