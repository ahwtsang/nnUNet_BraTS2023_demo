"""nnU-Net v2 trainer: 250 epochs + Weights & Biases experiment tracking.

What this adds on top of the stock ``nnUNetTrainer``:
  * ``num_epochs = 250`` (down from the 1000 default) to keep cloud cost sane.
  * Per-epoch logging to W&B: train/val loss, per-region pseudo-Dice,
    EMA foreground Dice, learning rate and epoch wall-time.
  * One W&B run per fold, all grouped under the dataset name, with a
    deterministic run id so a preempted spot instance reconnects to the
    SAME run when training resumes (``resume="allow"``).

Logging is a no-op (training still works) if wandb is not installed, if
this is not the main DDP process, or if ``WANDB_DISABLED=true`` is set.

Install: copy this file into
  <site-packages>/nnunetv2/training/nnUNetTrainer/variants/
so that ``nnUNetv2_train ... -tr nnUNetTrainerWandb250`` can discover it.
(scripts/00_setup.sh does this automatically.)
"""

import os

import torch

from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer

try:
    import wandb
    _WANDB_AVAILABLE = True
except ImportError:  # training must still run without wandb installed
    _WANDB_AVAILABLE = False


class nnUNetTrainerWandb250(nnUNetTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # ---- the only hyper-parameter change vs. the default trainer ----
        self.num_epochs = 250
        self._wandb_run = None

    # ------------------------------------------------------------------ #
    # helpers
    # ------------------------------------------------------------------ #
    def _wandb_enabled(self) -> bool:
        return (
            _WANDB_AVAILABLE
            and getattr(self, "local_rank", 0) == 0
            and os.environ.get("WANDB_DISABLED", "false").lower() != "true"
        )

    def _dataset_name(self) -> str:
        try:
            return self.plans_manager.dataset_name
        except Exception:
            return os.path.basename(getattr(self, "preprocessed_dataset_folder_base", "dataset"))

    def _region_names(self):
        """Foreground label/region names from dataset.json, in order."""
        try:
            return [k for k in self.dataset_json["labels"].keys() if k != "background"]
        except Exception:
            return None

    @staticmethod
    def _last(log: dict, key):
        vals = log.get(key, [])
        return vals[-1] if len(vals) else None

    # ------------------------------------------------------------------ #
    # lifecycle hooks
    # ------------------------------------------------------------------ #
    def on_train_start(self):
        super().on_train_start()
        if not self._wandb_enabled():
            return
        ds_name = self._dataset_name()
        run_id = f"{ds_name}_{self.configuration_name}_fold{self.fold}"
        self._wandb_run = wandb.init(
            project=os.environ.get("WANDB_PROJECT", "nnunet-brats2023-5fold"),
            entity=os.environ.get("WANDB_ENTITY"),  # None -> default entity
            group=ds_name,            # the 5 folds share one group
            job_type="train",
            name=run_id,
            id=run_id,                # deterministic -> resume reconnects
            resume="allow",
            config={
                "dataset": ds_name,
                "configuration": self.configuration_name,
                "fold": self.fold,
                "num_epochs": self.num_epochs,
                "batch_size": self.batch_size,
                "patch_size": list(self.configuration_manager.patch_size),
                "initial_lr": self.initial_lr,
                "weight_decay": self.weight_decay,
                "trainer": type(self).__name__,
            },
        )

    def on_epoch_end(self):
        # Let nnU-Net do its own console logging / checkpointing first.
        super().on_epoch_end()
        if not self._wandb_enabled() or self._wandb_run is None:
            return

        log = self.logger.my_fantastic_logging
        epoch = self.current_epoch

        metrics = {
            "train/loss": self._last(log, "train_losses"),
            "val/loss": self._last(log, "val_losses"),
            "val/ema_fg_dice": self._last(log, "ema_fg_dice"),
            "val/mean_fg_dice": self._last(log, "mean_fg_dice"),
            "lr": self._last(log, "lrs"),
        }

        starts = log.get("epoch_start_timestamps", [])
        ends = log.get("epoch_end_timestamps", [])
        if len(starts) and len(ends):
            metrics["time/epoch_seconds"] = ends[-1] - starts[-1]

        # per-region pseudo-Dice (BraTS: whole_tumor / tumor_core / enhancing_tumor)
        dice = self._last(log, "dice_per_class_or_region")
        if dice is not None:
            names = self._region_names()
            for i, d in enumerate(dice):
                label = names[i] if names and i < len(names) else f"region_{i}"
                metrics[f"val/dice_{label}"] = float(d)

        metrics = {k: v for k, v in metrics.items() if v is not None}
        wandb.log(metrics, step=epoch)

    def on_train_end(self):
        super().on_train_end()
        if self._wandb_enabled() and self._wandb_run is not None:
            wandb.finish()
            self._wandb_run = None
