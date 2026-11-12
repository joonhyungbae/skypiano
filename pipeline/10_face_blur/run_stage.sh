#!/bin/bash
#=============================================================================
# Stage 11: EgoBlur Video Anonymization
#=============================================================================
# Usage:
#   ./run_stage.sh                         # blur DEFAULT_SCOPE (240828, all 4 angles)
#   ./run_stage.sh --only 240828           # one dataset
#   ./run_stage.sh --only 240828 --session C0_N4
#   ./run_stage.sh --angle Left            # one camera angle
#   ./run_stage.sh --force                 # re-encode even if outputs exist
#
# Note: 240809 is excluded from the public release-blur scope (see
# DEFAULT_SCOPE in steps/s01_blur_videos.py); only 240828 is processed
# by default.
#
# Environment:
#   STEPS                    steps to run (default: s01)
#   EGOBLUR_FACE_THRESHOLD   detection threshold (default 0.9)
#   EGOBLUR_NMS_IOU          NMS IoU threshold  (default 0.3)
#   EGOBLUR_BLUR_SIGMA_K     blur sigma multiplier (default 0.35)
#   EGOBLUR_FRAME_STRIDE     detect every N frames (default 1)
#   EGOBLUR_DEVICE           cuda | cpu (default auto)
#=============================================================================
set -euo pipefail
cd "$(dirname "$0")"

ARGS="$@"
STEPS="${STEPS:-s01}"

echo "================================================================"
echo "Stage 11: EgoBlur Video Anonymization"
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
    s01) run_module "s01 blur videos"  steps.s01_blur_videos ;;
    *) echo "Unknown step: $step"; exit 1 ;;
  esac
done

echo ""
echo "================================================================"
echo "Stage 11 complete. Output: Data/Pipeline/10_face_blur/"
echo "================================================================"
