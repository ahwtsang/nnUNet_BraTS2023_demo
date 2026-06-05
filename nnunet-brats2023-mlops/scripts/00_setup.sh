#!/usr/bin/env bash
# Step 0 - install dependencies and register the custom W&B trainer.
# Run once per fresh pod:  bash scripts/00_setup.sh
set -euo pipefail
cd "$(dirname "$0")/.."

pip install --upgrade pip
pip install -r requirements.txt

# nnU-Net discovers trainers by scanning its own package directory, so the
# custom trainer file has to live inside the installed nnunetv2 package.
TRAINER_DIR=$(python -c "import nnunetv2, os; print(os.path.join(os.path.dirname(nnunetv2.__file__), 'training', 'nnUNetTrainer', 'variants'))")
mkdir -p "$TRAINER_DIR"
cp nnunet_trainer/nnUNetTrainerWandb250.py "$TRAINER_DIR/"
echo "Installed custom trainer -> $TRAINER_DIR/nnUNetTrainerWandb250.py"

python - <<'PY'
import torch
print("torch:", torch.__version__, "| CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
PY

echo "Setup complete. Next: source env.sh && wandb login"
