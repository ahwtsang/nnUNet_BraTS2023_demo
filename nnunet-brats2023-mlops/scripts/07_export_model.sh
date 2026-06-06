#!/usr/bin/env bash
# Step 7 - package the trained model into a single portable .zip.
#
# Training already saved checkpoint_final.pth for each fold under
# nnUNet_results. This bundles the plans, dataset.json and all 5 fold
# checkpoints into one file you can attach to a GitHub Release or import
# on another machine. This is the idiomatic nnU-Net "model file".
#
# Usage:  bash scripts/07_export_model.sh [DATASET_ID] [OUTPUT_ZIP]
set -euo pipefail

DATASET_ID=${1:-137}
OUT_ZIP=${2:-nnunet_brats2023_3dfullres_model.zip}
TRAINER=nnUNetTrainerWandb250

: "${nnUNet_results:?Set nnU-Net env vars first: source env.sh}"

nnUNetv2_export_model_to_zip \
    -d "$DATASET_ID" \
    -o "$OUT_ZIP" \
    -c 3d_fullres \
    -tr "$TRAINER" \
    -f 0 1 2 3 4 \
    -chk checkpoint_final.pth

echo "Exported model -> $OUT_ZIP"
echo "Import on another machine with:"
echo "    nnUNetv2_install_pretrained_model_from_zip $OUT_ZIP"
