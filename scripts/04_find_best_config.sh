#!/usr/bin/env bash
# Step 4 - evaluate the 5-fold ensemble and determine postprocessing.
# Requires that training was run with --npz (scripts/03 does this).
# Writes the cross-validation summary + inference instructions under
# nnUNet_results. Usage:  bash scripts/04_find_best_config.sh [DATASET_ID]
set -euo pipefail

DATASET_ID=${1:-137}
TRAINER=nnUNetTrainerWandb250
PLANS=nnUNetResEncUNetMPlans

nnUNetv2_find_best_configuration "$DATASET_ID" -c 3d_fullres -tr "$TRAINER" -p "$PLANS"

echo "Cross-validation summary and postprocessing written under nnUNet_results."
echo "See the generated inference_instructions.txt for the predict command."
