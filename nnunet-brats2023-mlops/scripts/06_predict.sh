#!/usr/bin/env bash
# Step 6 - run inference with the trained 5-fold ensemble.
#
# Takes a folder of BraTS-style case subfolders, prepares nnU-Net inputs,
# runs the 5-fold ensemble, and applies the postprocessing determined by
# scripts/04_find_best_config.sh (if present).
#
# Usage:
#   bash scripts/06_predict.sh <brats_cases_dir> <output_dir> [DATASET_ID] [DATASET_NAME]
set -euo pipefail
cd "$(dirname "$0")/.."

BRATS_DIR=${1:?Usage: 06_predict.sh <brats_cases_dir> <output_dir> [DATASET_ID] [DATASET_NAME]}
OUT_DIR=${2:?Usage: 06_predict.sh <brats_cases_dir> <output_dir> [DATASET_ID] [DATASET_NAME]}
DATASET_ID=${3:-137}
DATASET_NAME=${4:-BraTS2023}
CONFIG=3d_fullres
TRAINER=nnUNetTrainerWandb250
PLANS=nnUNetPlans

: "${nnUNet_results:?Set nnU-Net env vars first: source env.sh}"

INFER_IN="${OUT_DIR}/_nnunet_inputs"
RAW_OUT="${OUT_DIR}/predictions"
PP_OUT="${OUT_DIR}/predictions_postprocessed"
mkdir -p "$INFER_IN" "$RAW_OUT"

echo "[1/3] Preparing inference inputs (channel naming)..."
python scripts/prepare_inference_inputs.py --brats_dir "$BRATS_DIR" --out_dir "$INFER_IN"

echo "[2/3] Running 5-fold ensemble inference..."
nnUNetv2_predict \
    -i "$INFER_IN" \
    -o "$RAW_OUT" \
    -d "$DATASET_ID" \
    -c "$CONFIG" \
    -tr "$TRAINER" \
    -p "$PLANS" \
    -f 0 1 2 3 4 \
    -chk checkpoint_final.pth

echo "[3/3] Applying postprocessing (if available)..."
DS_DIR=$(printf 'Dataset%03d_%s' "$DATASET_ID" "$DATASET_NAME")
PP_PKL=$(find "${nnUNet_results}/${DS_DIR}" -name postprocessing.pkl 2>/dev/null | head -n 1 || true)

if [ -n "${PP_PKL}" ] && [ -f "${PP_PKL}" ]; then
    mkdir -p "$PP_OUT"
    # nnUNetv2_predict writes plans.json + dataset.json into the output folder,
    # so we reuse those for postprocessing.
    nnUNetv2_apply_postprocessing \
        -i "$RAW_OUT" \
        -o "$PP_OUT" \
        --pp_pkl_file "$PP_PKL" \
        -plans_json "${RAW_OUT}/plans.json" \
        -dataset_json "${RAW_OUT}/dataset.json" \
        -np 8
    echo "Done.  raw -> ${RAW_OUT}   postprocessed -> ${PP_OUT}"
else
    echo "No postprocessing.pkl found - run scripts/04_find_best_config.sh first to generate it."
    echo "Done.  predictions -> ${RAW_OUT}"
fi
