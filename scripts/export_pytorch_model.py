#!/usr/bin/env python
"""Create a standalone PyTorch model file from one trained nnU-Net fold.

Produces:
  1) <out>.pth     - a clean checkpoint containing ONLY the network weights
                     plus minimal metadata (input channels, patch size, region
                     names). Loadable with plain torch.load - no nnU-Net needed
                     just to inspect or load the weights.
  2) <out>.ts.pt   - (optional, --torchscript) a TorchScript trace of the
                     network for framework-agnostic deployment.

IMPORTANT: the raw network expects ALREADY-PREPROCESSED input (per-channel
z-scored, resampled to the training spacing, tiled to the patch size). nnU-Net's
preprocessing and sliding-window logic are NOT inside these files. For real
end-to-end inference on raw MRI use scripts/06_predict.sh or the exported model
zip (scripts/07_export_model.sh). These exports exist to demonstrate a portable
PyTorch artifact / a custom deployment path.

Usage:
    python scripts/export_pytorch_model.py --fold 0 --out model_brats2023_fold0
    python scripts/export_pytorch_model.py --fold 0 --torchscript --device cuda
"""
import argparse
import os
from pathlib import Path

import torch
from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results_dir", default=os.environ.get("nnUNet_results"))
    ap.add_argument("--dataset", default="Dataset137_BraTS2023")
    ap.add_argument("--trainer", default="nnUNetTrainerWandb250")
    ap.add_argument("--config", default="3d_fullres")
    ap.add_argument("--plans", default="nnUNetResEncUNetMPlans")
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--checkpoint", default="checkpoint_final.pth")
    ap.add_argument("--out", default="model_brats2023_fold0")
    ap.add_argument("--torchscript", action="store_true")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    assert args.results_dir, "Set nnUNet_results (source env.sh) or pass --results_dir"
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    model_folder = Path(args.results_dir) / args.dataset / f"{args.trainer}__{args.plans}__{args.config}"
    assert model_folder.is_dir(), f"Model folder not found: {model_folder}"

    # nnUNetPredictor rebuilds the exact architecture and loads the checkpoint,
    # so we don't have to reconstruct the network from plans by hand.
    predictor = nnUNetPredictor(device=device, allow_tqdm=False)
    predictor.initialize_from_trained_model_folder(
        str(model_folder), use_folds=(args.fold,), checkpoint_name=args.checkpoint
    )

    # unwrap torch.compile if present, and put in single-output inference mode
    net = getattr(predictor.network, "_orig_mod", predictor.network)
    net.eval()
    if hasattr(net, "deep_supervision"):
        net.deep_supervision = False

    num_input_channels = len(predictor.dataset_json["channel_names"])
    patch_size = list(predictor.configuration_manager.patch_size)
    region_names = [k for k in predictor.dataset_json["labels"] if k != "background"]

    # ---- 1) weights-only checkpoint -----------------------------------------
    pth_path = f"{args.out}.pth"
    torch.save(
        {
            "network_weights": net.state_dict(),
            "num_input_channels": num_input_channels,
            "patch_size": patch_size,
            "region_names": region_names,
            "regions_class_order": predictor.dataset_json.get("regions_class_order"),
            "trainer": args.trainer,
            "configuration": args.config,
            "fold": args.fold,
            "source_checkpoint": args.checkpoint,
        },
        pth_path,
    )
    print(f"Wrote weights-only checkpoint -> {pth_path}")
    print(f"  input_channels={num_input_channels}  patch_size={patch_size}  regions={region_names}")

    # ---- 2) optional TorchScript --------------------------------------------
    if args.torchscript:
        example = torch.zeros(1, num_input_channels, *patch_size, device=device)
        try:
            with torch.no_grad():
                traced = torch.jit.trace(net, example)
            ts_path = f"{args.out}.ts.pt"
            traced.save(ts_path)
            print(f"Wrote TorchScript model -> {ts_path}")
        except Exception as e:  # some architectures with dynamic control flow won't trace
            print(f"TorchScript trace failed ({type(e).__name__}: {e}).")
            print("The weights-only .pth above is still valid; skip --torchscript if not needed.")


if __name__ == "__main__":
    main()
