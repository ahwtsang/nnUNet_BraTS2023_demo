# nnU-Net BraTS 2023 — 5-Fold Cross-Validation on RunPod (RTX 4090) with W&B

A reproducible MLOps portfolio project: train **nnU-Net v2** (`3d_fullres`,
region-based) for brain-tumor segmentation on **BraTS 2023 Adult Glioma**, run a
proper **5-fold cross-validation at 250 epochs/fold**, track everything in
**Weights & Biases**, all on a single **RTX 4090** rented from **RunPod**.

```
nnunet-brats2023-mlops/
├── env.sh                          # nnU-Net paths + W&B config (source it)
├── requirements.txt
├── Dockerfile                      # optional, for a pinned environment
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
│   └── export_pytorch_model.py     # standalone .pth / TorchScript export
└── blog/writeup.md                 # blog-post skeleton
```

---

## Step 1 — Create the RunPod pod

1. **GPU:** RTX 4090 (24 GB). Standard `3d_fullres` for BraTS fits comfortably,
   so you do not need an A100. Use **Community Cloud** for the cheapest rate, or
   tick **Spot / Interruptible** to go cheaper still — the training script is
   built to resume after preemption.
2. **Template:** any official **PyTorch** template (CUDA 12.x, PyTorch 2.x).
3. **Persistent storage — important:** attach a **Network Volume (~150 GB)**
   mounted at **`/workspace`**. BraTS raw is ~30–40 GB and preprocessing roughly
   doubles it. Putting everything on `/workspace` means your data, checkpoints
   and results survive pod stops and spot interruptions. (Network volumes are
   region-locked, so create the pod in the volume's region.)
4. **Access:** enable the Web Terminal (and/or SSH + Jupyter).

## Step 2 — Get the code and data onto the pod

```bash
cd /workspace
git clone <your-repo-url> project && cd project
```

BraTS 2023 is license-gated (register on Synapse), so it can't be auto-downloaded.
Get the training set onto the pod into a folder of per-case subfolders, e.g.
`/workspace/raw_brats2023/train/BraTS-GLI-XXXXX-XXX/...`. Options:

- `runpodctl send` from your machine, then `runpodctl receive` on the pod, or
- pull from your own cloud bucket with the provider CLI / `wget`.

## Step 3 — One-time environment setup

```bash
bash scripts/00_setup.sh        # installs nnU-Net + W&B, registers the trainer
source env.sh                   # sets nnUNet_raw / _preprocessed / _results
wandb login                     # paste your W&B API key (or set WANDB_API_KEY)
```

Edit `env.sh` first if you want to set `WANDB_ENTITY` or change the project name.
Re-run `source env.sh` at the start of every new session.

## Step 4 — Convert BraTS → nnU-Net format

```bash
python scripts/01_convert_dataset.py --brats_dir /workspace/raw_brats2023/train
```

This writes `Dataset137_BraTS2023` with channels `_0000.._0003`
(T1 / T1ce / T2 / FLAIR), region-based labels (WT / TC / ET), and a
`dataset.json`. It auto-remaps enhancing-tumor label `4 → 3` if your copy uses
the older encoding.

## Step 5 — Plan and preprocess

```bash
bash scripts/02_preprocess.sh 137
```

Fingerprinting + preprocessing for `3d_fullres` only (skips 2d / 3d_lowres to
save time and disk). Expect roughly 1–3 hours, mostly CPU-bound.

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
time is roughly constant. On a 4090 figure ~2 min/epoch → ~8 hr/fold →
**~40 GPU-hours** for all five folds. At Community-Cloud RTX 4090 rates
(~$0.34/hr) that's **~$15**, plus a few dollars of preprocessing/storage; spot
pricing lands it under $10. *(Benchmark a few epochs first — the rate from
`wandb` `time/epoch_seconds` gives you an exact projection.)*

## Step 7 — Cross-validation summary

```bash
bash scripts/04_find_best_config.sh 137      # CV ensemble + postprocessing
python scripts/05_aggregate_wandb.py         # mean ± std Dice -> W&B summary
```

`05_aggregate_wandb.py` reads each fold's `validation/summary.json` and logs a
per-fold/per-region table plus mean ± std Dice (WT / TC / ET) and an overall
mean to a dedicated `cv_summary` W&B run — the centerpiece chart for your blog.

## Step 8 — Run inference

Predict on new / held-out cases with the full 5-fold ensemble. Input is a folder
of BraTS-style case subfolders (each holding the 4 modality `.nii.gz` files; a
seg is not needed for inference):

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

## Step 10 — Publish

- Commit code, configs and your W&B run links. **Do not commit data or weights**
  (`.gitignore` already excludes `*.nii.gz`, `*.pth`, `nnUNet_*`, `wandb/`).
- The model `.zip` can be large — attach it to a **GitHub Release** or use
  **Git LFS** rather than committing it into history.
- Flesh out `blog/writeup.md`, embedding W&B charts and a prediction-vs-ground-
  truth overlay.

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

### Notes & knobs
- **Fewer/more epochs:** change `self.num_epochs` in `nnUNetTrainerWandb250.py`
  (re-run `00_setup.sh` to reinstall), then retrain.
- **Disable tracking:** `export WANDB_DISABLED=true` (training still runs).
- **Bigger model later:** add ResEnc planning
  (`nnUNetv2_plan_and_preprocess -d 137 -pl nnUNetPlannerResEncM -c 3d_fullres`)
  and train with `-p nnUNetResEncUNetMPlans`. ResEncM/L fit a 4090.
- Tune `nnUNet_n_proc_DA` in `env.sh` to your pod's vCPU count (16–18 for a 4090).
