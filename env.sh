# Source this file at the start of every session:  source env.sh
#
# On RunPod, /workspace is the persistent network volume, so everything
# below survives pod restarts and spot-instance interruptions.

export nnUNet_raw="/workspace/nnUNet_raw"
export nnUNet_preprocessed="/workspace/nnUNet_preprocessed"
export nnUNet_results="/workspace/nnUNet_results"
mkdir -p "$nnUNet_raw" "$nnUNet_preprocessed" "$nnUNet_results"

# Data-augmentation worker processes. 16-18 is the recommended value for an
# RTX 4090 (raise/lower to match the vCPU count of your pod).
export nnUNet_n_proc_DA=16

# ---- Weights & Biases ----
export WANDB_PROJECT="nnunet-brats2023-5fold"
# export WANDB_ENTITY="your-wandb-username-or-team"   # optional
# Authenticate once per pod, either by running `wandb login` or by setting:
export WANDB_API_KEY="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"
# To turn tracking off entirely (training still runs): export WANDB_DISABLED=true

echo "nnU-Net paths set. raw=$nnUNet_raw  preprocessed=$nnUNet_preprocessed  results=$nnUNet_results"
