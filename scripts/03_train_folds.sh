#!/usr/bin/env bash
# Step 3 - run 5-fold cross-validation, 250 epochs per fold, with W&B tracking.
#
# Spot-instance friendly: re-running this script skips completed folds and
# resumes an interrupted fold from its latest checkpoint. So if a spot pod is
# reclaimed mid-run, just start a new pod, `source env.sh`, and run this again.
#
# Usage:  bash scripts/03_train_folds.sh [DATASET_ID] [DATASET_NAME]
set -euo pipefail

DATASET_ID=${1:-137}
DATASET_NAME=${2:-BraTS2023}
CONFIG=3d_fullres
TRAINER=nnUNetTrainerWandb250
PLANS=nnUNetResEncUNetMPlans

: "${nnUNet_results:?Set nnU-Net env vars first: source env.sh}"

DS_DIR=$(printf 'Dataset%03d_%s' "$DATASET_ID" "$DATASET_NAME")

for FOLD in 0 1 2 3 4; do
  OUT="${nnUNet_results}/${DS_DIR}/${TRAINER}__${PLANS}__${CONFIG}/fold_${FOLD}"

  if [ -f "${OUT}/checkpoint_final.pth" ]; then
    echo "=== Fold ${FOLD}: already complete, skipping ==="
    continue
  fi

  if [ -f "${OUT}/checkpoint_latest.pth" ]; then
    echo "=== Fold ${FOLD}: resuming from latest checkpoint ==="
    nnUNetv2_train "$DATASET_ID" "$CONFIG" "$FOLD" -tr "$TRAINER" -p "$PLANS" --npz --c
  else
    echo "=== Fold ${FOLD}: starting fresh ==="
    nnUNetv2_train "$DATASET_ID" "$CONFIG" "$FOLD" -tr "$TRAINER"  -p "$PLANS" --npz
  fi
done

echo "All 5 folds finished."
