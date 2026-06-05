#!/usr/bin/env bash
# Step 2 - fingerprint, plan and preprocess the dataset.
# Restricted to 3d_fullres to save time and disk (skips 2d / 3d_lowres).
# Usage:  bash scripts/02_preprocess.sh [DATASET_ID]
set -euo pipefail

DATASET_ID=${1:-137}
: "${nnUNet_preprocessed:?Set nnU-Net env vars first: source env.sh}"

nnUNetv2_plan_and_preprocess \
    -d "$DATASET_ID" \
    -c 3d_fullres \
    --verify_dataset_integrity \
    -pl nnUNetPlannerResEncM

echo "Preprocessing complete for dataset $DATASET_ID (3d_fullres)."
