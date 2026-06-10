# nnU-Net BraTS 2023 — 5-Fold Cross-Validation on RunPod (RTX PRO 4500 Blackwell) with W&B

A reproducible MLOps portfolio project: train **nnU-Net v2** (`3d_fullres`,
region-based) for brain-tumor segmentation on **BraTS 2023 Adult Glioma**, run a
proper **5-fold cross-validation at 250 epochs/fold**, track everything in
**Weights & Biases**, all on a single **RTX PRO 4500 Blackwell** rented from **RunPod**.

```
nnUNet_BraTS2023_demo/
├── env.sh                          # nnU-Net paths + W&B config (source it)
├── requirements.txt
├── nnunet_trainer/
│   └── nnUNetTrainerWandb250.py    # custom trainer: 250 epochs + W&B logging
├── scripts/
│   ├── 00_setup.sh                 # install deps + register the trainer
│   ├── 01_convert_dataset.py       # BraTS 2023 -> nnU-Net raw format
│   ├── 02_preprocess.sh            # plan + preprocess (3d_fullres)
│   ├── 03_train_folds.sh           # 5 folds x 250 epochs, resume-aware
│   ├── 04_find_best_config.sh      # CV evaluation + postprocessing
│   ├── 05_aggregate_wandb.py       # mean ± std Dice summary -> W&B
│   ├── prepare_inference_inputs.py # BraTS cases -> channel-named inputs
│   ├── 06_predict.sh               # 5-fold ensemble inference
│   ├── 07_export_model.sh          # package the shareable model .zip
│   ├── export_pytorch_model.py     # standalone .pth / TorchScript export
│   └── overlay_prediction.py       # prediction overlay PNG
├── Dockerfile                      # for a pinned environment
├── Dockerfile.inference            # inference-only image (bakes in model .zip)
├── docker/
│   └── entrypoint_predict.sh       # entrypoint for the inference image
```

---

## Step 1 — Create the RunPod pod

1. **GPU:** RTX PRO 4500 Blackwell (32 GB). Standard `3d_fullres` for BraTS fits
   comfortably.
2. **Template:** any official **PyTorch** template (CUDA 12.x, PyTorch 2.x).
   For this project, the template runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04 was used
3. **Persistent storage — important:** attach a **Network Volume (100 GB)**
   mounted at **`/workspace`**. BraTS raw is ~30–40 GB and preprocessing roughly
   doubles it. Putting everything on `/workspace` means data, checkpoints
   and results survive pod stops and spot interruptions. (Network volumes are
   region-locked, so the pod needs to be created in the volume's region.)
4. **Access:** enable the Web Terminal (and SSH).

## Step 2 — Get the code and data onto the pod

```bash
cd /workspace
git clone <your-repo-url> project && cd project
```

The BraTS 2023 datasets are available to all registered Synapse users who accept the post-challenge terms and conditions. The training and validation sets were downloaded from Synapse and rsync to get the data to the pod. The training dataset contains 1251 cases, each case is a subfolder that contains 4 modality images and segmentation as:
```
├── BraTS-GLI-00000-000/
│   ├──BraTS-GLI-00000-000-t1c.nii.gz
│   ├──BraTS-GLI-00000-000-t1n.nii.gz
│   ├──BraTS-GLI-00000-000-t2f.nii.gz
│   ├──BraTS-GLI-00000-000-t2w.nii.gz
│   ├──BraTS-GLI-00000-000-seg.nii.gz
```
The validation dataset contains 219 cases with similar folder structure as the training dataset.

## Step 3 — One-time environment setup

```bash
bash scripts/00_setup.sh        # installs nnU-Net + W&B, registers the trainer
source env.sh                   # sets nnUNet_raw / _preprocessed / _results
wandb login                     # paste your W&B API key (or set WANDB_API_KEY)
```

Add W&B API key to `WANDB_API_KEY` in `env.sh`.
Re-run `source env.sh` at the start of every new session.

## Step 4 — Convert BraTS → nnU-Net format

```bash
python scripts/01_convert_dataset.py --brats_dir /workspace/raw_brats2023/train
```

This writes `Dataset137_BraTS2023` with channels `_0000.._0003`
(T1 / T1ce / T2 / FLAIR), region-based labels (WT / TC / ET), and a
`dataset.json`. It auto-remaps enhancing-tumor label `4 → 3` if the data version uses the older encoding.

## Step 5 — Plan and preprocess

```bash
bash scripts/02_preprocess.sh 137
```

Fingerprinting + preprocessing for `3d_fullres` only (skips 2d / 3d_lowres to
save time and disk). Takes roughly 1–3 hours, mostly CPU-bound.

## Step 6 — Train the 5 folds (250 epochs each)

```bash
bash scripts/03_train_folds.sh 137 BraTS2023
```

- Trains folds 0–4 with `nnUNetTrainerWandb250` (250 epochs, `--npz` for later
  ensembling).
- **Resume-aware:** completed folds are skipped; an interrupted fold continues
  from its latest checkpoint. If a spot pod is reclaimed, start a new pod,
  `source env.sh`, and run this same command again.
- Each fold streams to W&B as its own run, grouped under `Dataset137_BraTS2023`,
  with a deterministic run id so a resumed fold reconnects to the same run.

**Rough timing/cost:** nnU-Net runs a fixed 250 iterations/epoch, so per-epoch
time is roughly constant. On the RTX PRO 4500, timing is approx. 50s/epoch, 4 hr/fold,
**20 GPU-hours** for all five folds. The RunPod rate for RTX PRO 4500 is
($0.74/hr) so total cost is approx. **$15** plus a few dollars of preprocessing and storage.

## Step 7 — Cross-validation summary

```bash
bash scripts/04_find_best_config.sh 137      # CV ensemble + postprocessing
python scripts/05_aggregate_wandb.py         # mean ± std Dice -> W&B summary
```

`05_aggregate_wandb.py` reads each fold's `validation/summary.json` and logs a
per-fold/per-region table plus mean ± std Dice (WT / TC / ET) and an overall
mean to a dedicated `cv_summary` W&B run.

## Step 8 — Run inference

Predict on new / held-out cases with the full 5-fold ensemble. Input is a folder
of BraTS-style case subfolders (each holding the 4 modality `.nii.gz` files; a
seg is not needed for inference).

```bash
bash scripts/06_predict.sh /workspace/raw_brats2023/val /workspace/preds 137 BraTS2023
```

This (1) copies each case's modalities into nnU-Net channel naming
(`CASEID_0000.._0003`), (2) runs `nnUNetv2_predict` with `-f 0 1 2 3 4` so all
five folds are ensembled, then (3) applies the postprocessing from Step 7 if its
`postprocessing.pkl` exists. Outputs land in `…/predictions/` (and
`…/predictions_postprocessed/`).

## Step 9 — Create the model file

Training already wrote a PyTorch checkpoint (`checkpoint_final.pth`) for **each
fold** under `nnUNet_results`. This step packages those into portable artifacts.

**Shareable nnU-Net model (recommended)** — bundles plans, `dataset.json` and all
five fold checkpoints into one `.zip` you can attach to a GitHub Release:

```bash
bash scripts/07_export_model.sh 137 nnunet_brats2023_3dfullres_model.zip
# import elsewhere:  nnUNetv2_install_pretrained_model_from_zip <file>.zip
```

**Standalone PyTorch file (optional)** — a clean weights-only `.pth`, plus an
optional TorchScript `.ts.pt`, built straight from a fold via `nnUNetPredictor`:

```bash
python scripts/export_pytorch_model.py --fold 0 --out model_brats2023_fold0 --torchscript
```

> Note: the standalone files contain only the network weights/graph. nnU-Net's
> preprocessing and sliding-window inference are **not** inside them, so for real
> end-to-end inference on raw MRI use Step 8 or the exported `.zip`. The
> standalone export is for demonstrating a framework-agnostic deployment artifact.

## Step 10 — Prediction visual: prediction overlay over FLAIR or T1CE

Render the figure to show — MRI, prediction, ground truth (if available)
side by side on the most informative slice.

```bash
python scripts/overlay_prediction.py \
    --image /workspace/preds/_nnunet_inputs/<CASEID>_0003.nii.gz \
    --pred  /workspace/preds/predictions/<CASEID>.nii.gz \
    --gt    /workspace/nnUNet_raw/Dataset137_BraTS2023/labelsTr/<CASEID>.nii.gz (optional) \
    --out   overlay_<CASEID>.png
```

Use channel `_0003` (FLAIR) as the background to show the whole tumor, or `_0001`
(T1ce) to emphasize the enhancing core. The script derives the three nested
regions correctly from each source: the ground truth from its sub-region labels
(NCR/ED/ET) and the prediction from nnU-Net's ordinal region encoding (≥1 WT,
≥2 TC, ≥3 ET), so the panels are directly comparable. Drop `--gt` for inference
cases with no ground truth. Note: The BraTS-GLI 2023 Validation dataset does not 
include the segmentation ground truth.

## Step 11 — Inference-only Docker image

Package the exported model (`.zip` from Step 9) into a self-contained image that
runs prediction with no setup.

```bash
# build (model zip must be in the build context)
docker build -f Dockerfile.inference \
    --build-arg MODEL_ZIP=nnunet_brats2023_3dfullres_model.zip \
    -t brats-nnunet-infer .

# run: mount a folder of BraTS cases and an output folder
docker run --gpus all -v /path/to/cases:/input -v /path/to/out:/output brats-nnunet-infer
```

The image installs the model with `nnUNetv2_install_pretrained_model_from_zip`
at build time; the entrypoint auto-prepares channel-named inputs and runs the
5-fold ensemble. `WANDB_DISABLED=true` is set so nothing tries to log.

## Step 12 — Publish

- Images and weights are **not** committed to the repository, i.e. 
  `.gitignore` excludes `*.nii.gz`, `*.pth`, `nnUNet_*`, `wandb/`.
- W&B link for the 5-fold training and cross-validation summary: https://wandb.ai/adrianhwtsang-none/nnunet-brats2023-5fold
- The model `.zip` is large — added in parts to **model_storage** repo (https://github.com/ahwtsang/model_storage).

---

### Quick reference

| Step | Command |
|------|---------|
| Setup | `bash scripts/00_setup.sh && source env.sh && wandb login` |
| Convert | `python scripts/01_convert_dataset.py --brats_dir <dir>` |
| Preprocess | `bash scripts/02_preprocess.sh 137` |
| Train (5 folds) | `bash scripts/03_train_folds.sh 137 BraTS2023` |
| CV summary | `bash scripts/04_find_best_config.sh 137` |
| Aggregate to W&B | `python scripts/05_aggregate_wandb.py` |
| Inference | `bash scripts/06_predict.sh <cases_dir> <out_dir> 137 BraTS2023` |
| Export model .zip | `bash scripts/07_export_model.sh 137 model.zip` |
| Standalone .pth | `python scripts/export_pytorch_model.py --fold 0 --out model_fold0` |
| Overlay PNG | `python scripts/overlay_prediction.py --image <flair> --pred <pred> --gt <gt> --out fig.png` |
| Inference image | `docker build -f Dockerfile.inference -t brats-nnunet-infer .` |

### Notes & knobs
- **Fewer/more epochs:** change `self.num_epochs` in `nnUNetTrainerWandb250.py`
  (re-run `00_setup.sh` to reinstall), then retrain.
- **Disable tracking:** `export WANDB_DISABLED=true` (training still runs).
- Tune `nnUNet_n_proc_DA` in `env.sh` to the pod's vCPU count (24 for the RTX Pro 4500).
