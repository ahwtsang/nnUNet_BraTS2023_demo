#!/usr/bin/env python
"""Aggregate the 5-fold cross-validation Dice scores and push a summary to W&B.

Reads each fold's validation summary written by nnU-Net at:
  $nnUNet_results/<Dataset>/<trainer>__<plans>__<config>/fold_<k>/validation/summary.json

and logs to a single W&B run:
  * a per-fold / per-region table,
  * mean +/- std Dice per region (WT / TC / ET),
  * an overall mean Dice in the run summary.

Run AFTER all five folds have finished (and ideally after
scripts/04_find_best_config.sh).
"""
import argparse
import json
import os
from pathlib import Path

import numpy as np
import wandb

# regions_class_order was (1, 2, 3) in the dataset.json, so:
REGION_NAME = {"1": "whole_tumor", "2": "tumor_core", "3": "enhancing_tumor"}


def load_fold_dice(summary_path: Path) -> dict:
    """Return {region_name: dice} for one fold, parsed defensively."""
    with open(summary_path) as f:
        s = json.load(f)
    out = {}
    for k, v in s.get("mean", {}).items():
        name = REGION_NAME.get(str(k), str(k))
        if isinstance(v, dict) and "Dice" in v:
            out[name] = float(v["Dice"])
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results_dir", default=os.environ.get("nnUNet_results"))
    ap.add_argument("--dataset", default="Dataset137_BraTS2023")
    ap.add_argument("--trainer", default="nnUNetTrainerWandb250")
    ap.add_argument("--config", default="3d_fullres")
    ap.add_argument("--plans", default="nnUNetResEncUNetMPlans")
    args = ap.parse_args()

    assert args.results_dir, "Set nnUNet_results env var (source env.sh) or pass --results_dir"

    base = Path(args.results_dir) / args.dataset / f"{args.trainer}__{args.plans}__{args.config}"
    rows, per_region = [], {}

    for fold in range(5):
        sp = base / f"fold_{fold}" / "validation" / "summary.json"
        if not sp.exists():
            print(f"  ! missing {sp} (did fold {fold} finish?)")
            continue
        d = load_fold_dice(sp)
        for region, dice in d.items():
            rows.append([fold, region, dice])
            per_region.setdefault(region, []).append(dice)
        print(f"fold {fold}: {d}")

    if not rows:
        raise SystemExit("No fold summaries found - nothing to aggregate.")

    run = wandb.init(
        project=os.environ.get("WANDB_PROJECT", "nnunet-brats2023-5fold"),
        entity=os.environ.get("WANDB_ENTITY"),
        group=args.dataset,
        job_type="cv-summary",
        name="cv_summary",
    )

    run.log({"cv_dice_table": wandb.Table(columns=["fold", "region", "dice"], data=rows)})

    summary, all_means = {}, []
    for region, vals in per_region.items():
        m, sd = float(np.mean(vals)), float(np.std(vals))
        summary[f"cv/{region}_dice_mean"] = m
        summary[f"cv/{region}_dice_std"] = sd
        all_means.append(m)
        print(f"{region:>16}: {m:.4f} +/- {sd:.4f}  (n={len(vals)})")
    summary["cv/overall_dice_mean"] = float(np.mean(all_means))
    print(f"{'overall':>16}: {summary['cv/overall_dice_mean']:.4f}")

    run.summary.update(summary)
    run.finish()
    print("Logged cross-validation summary to W&B.")


if __name__ == "__main__":
    main()
