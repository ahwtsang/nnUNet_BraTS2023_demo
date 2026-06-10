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
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 device: torch.device = torch.device('cuda')):
        # NOTE: the signature must match nnUNetTrainer.__init__ EXACTLY (no
        # *args/**kwargs). The base __init__ introspects self.__init__'s
        # parameter names and looks each one up in locals(); a *args/**kwargs
        # signature makes it look for a local named "args" and raises
        # KeyError: 'args'.
        super().__init__(plans, configuration, fold, dataset_json, device)
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

    def _metric(self, key):
        """Latest value for a logged key, across nnU-Net logger versions.

        Newer nnU-Net uses MetaLogger with a public get_value(); older versions
        exposed an internal my_fantastic_logging dict. Returns None if neither
        is available so logging can never crash training.
        """
        logger = self.logger
        if hasattr(logger, "get_value"):
            try:
                return logger.get_value(key, step=-1)
            except Exception:
                return None
        store = getattr(logger, "my_fantastic_logging", None)
        if isinstance(store, dict):
            vals = store.get(key, [])
            return vals[-1] if len(vals) else None
        return None

    # ------------------------------------------------------------------ #
    # lifecycle hooks
    # ------------------------------------------------------------------ #
    def on_train_start(self):
        super().on_train_start()
        if not self._wandb_enabled():
            return
        ds_name = self._dataset_name()
        run_id = f"{ds_name}_{self.configuration_name}_fold{self.fold}"

        def _safe(fn):
            try:
                return fn()
            except Exception:
                return None

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
                "configuration": getattr(self, "configuration_name", None),
                "fold": self.fold,
                "num_epochs": self.num_epochs,
                "batch_size": getattr(self, "batch_size", None),
                "patch_size": _safe(lambda: list(self.configuration_manager.patch_size)),
                "initial_lr": getattr(self, "initial_lr", None),
                "weight_decay": getattr(self, "weight_decay", None),
                "trainer": type(self).__name__,
            },
        )

    def on_epoch_end(self):
        # Let nnU-Net do its own console logging / checkpointing first.
        super().on_epoch_end()
        if not self._wandb_enabled() or self._wandb_run is None:
            return

        epoch = self.current_epoch

        metrics = {
            "train/loss": self._metric("train_losses"),
            "val/loss": self._metric("val_losses"),
            "val/ema_fg_dice": self._metric("ema_fg_dice"),
            "val/mean_fg_dice": self._metric("mean_fg_dice"),
            "lr": self._metric("lrs"),
        }

        start = self._metric("epoch_start_timestamps")
        end = self._metric("epoch_end_timestamps")
        if start is not None and end is not None:
            metrics["time/epoch_seconds"] = end - start

        # per-region pseudo-Dice (BraTS: whole_tumor / tumor_core / enhancing_tumor)
        dice = self._metric("dice_per_class_or_region")
        if dice is not None:
            names = self._region_names()
            for i, d in enumerate(dice):
                label = names[i] if names and i < len(names) else f"region_{i}"
                metrics[f"val/dice_{label}"] = float(d)

        metrics = {k: v for k, v in metrics.items() if v is not None}
        if metrics:
            wandb.log(metrics, step=epoch)

    def on_train_end(self):
        super().on_train_end()
        if self._wandb_enabled() and self._wandb_run is not None:
            wandb.finish()
            self._wandb_run = None
