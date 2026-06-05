# Optional: pin a reproducible environment. Match the tag to an available
# RunPod PyTorch image (check the RunPod template list for the current tag).
FROM runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04

WORKDIR /workspace/project
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# register the custom trainer inside the nnU-Net package
RUN TRAINER_DIR=$(python -c "import nnunetv2, os; print(os.path.join(os.path.dirname(nnunetv2.__file__), 'training', 'nnUNetTrainer', 'variants'))") \
    && cp nnunet_trainer/nnUNetTrainerWandb250.py "$TRAINER_DIR/"
