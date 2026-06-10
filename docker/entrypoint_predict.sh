#!/usr/bin/env bash
# Entrypoint for the inference-only image. Reads BraTS cases from $INPUT and
# writes segmentations to $OUTPUT using the model baked into the image.
#
#   INPUT  : either a folder of BraTS case subfolders (auto-prepared), or a
#            folder already in nnU-Net channel naming (CASEID_0000..0003.nii.gz)
#   OUTPUT : folder for predicted .nii.gz masks
set -euo pipefail

INPUT=${INPUT:-/input}
OUTPUT=${OUTPUT:-/output}
DATASET_ID=${DATASET_ID:-137}
TRAINER=${TRAINER:-nnUNetTrainerWandb250}
CONFIG=${CONFIG:-3d_fullres}

mkdir -p "$OUTPUT"

# If INPUT contains subdirectories, treat them as raw BraTS cases and prepare
# channel-named inputs; otherwise assume INPUT is already nnU-Net formatted.
if find "$INPUT" -mindepth 1 -maxdepth 1 -type d | read -r _; then
    PREP=/tmp/nnunet_inputs
    mkdir -p "$PREP"
    python /app/prepare_inference_inputs.py --brats_dir "$INPUT" --out_dir "$PREP"
    IN="$PREP"
else
    IN="$INPUT"
fi

nnUNetv2_predict \
    -i "$IN" \
    -o "$OUTPUT" \
    -d "$DATASET_ID" \
    -c "$CONFIG" \
    -tr "$TRAINER" \
    -f 0 1 2 3 4 \
    -chk checkpoint_final.pth

echo "Inference complete -> $OUTPUT"
