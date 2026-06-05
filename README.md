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
│   └── 05_aggregate_wandb.py       # mean ± std Dice summary -> W&B
└── blog/writeup.md                 # blog-post skeleton
```

---

## Step 1 — Create the RunPod pod

1. **GPU:** RTX 4090 (24 GB). Standard `3d_fullres` for BraTS fits comfortably.
   Use RunPod **Community Cloud** for the cheapest rate, or
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

## Step 8 — Publish

- Commit code, configs and your W&B run links. **Do not commit data or weights**
  (`.gitignore` already excludes `*.nii.gz`, `*.pth`, `nnUNet_*`, `wandb/`).
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

### Notes & knobs
- **Fewer/more epochs:** change `self.num_epochs` in `nnUNetTrainerWandb250.py`
  (re-run `00_setup.sh` to reinstall), then retrain.
- **Disable tracking:** `export WANDB_DISABLED=true` (training still runs).
- **Bigger model later:** add ResEnc planning
  (`nnUNetv2_plan_and_preprocess -d 137 -pl nnUNetPlannerResEncM -c 3d_fullres`)
  and train with `-p nnUNetResEncUNetMPlans`. ResEncM/L fit a 4090.
- Tune `nnUNet_n_proc_DA` in `env.sh` to your pod's vCPU count (16–18 for a 4090).
