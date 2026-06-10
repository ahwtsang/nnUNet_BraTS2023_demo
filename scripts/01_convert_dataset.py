#!/usr/bin/env python
"""Convert BraTS 2023 (Adult Glioma) into nnU-Net v2 raw format.

Expected input layout (one folder per case):

    <brats_dir>/BraTS-GLI-XXXXX-XXX/
        BraTS-GLI-XXXXX-XXX-t1n.nii.gz   native T1
        BraTS-GLI-XXXXX-XXX-t1c.nii.gz   contrast-enhanced T1
        BraTS-GLI-XXXXX-XXX-t2w.nii.gz   T2
        BraTS-GLI-XXXXX-XXX-t2f.nii.gz   T2-FLAIR
        BraTS-GLI-XXXXX-XXX-seg.nii.gz   labels

Output: $nnUNet_raw/Dataset137_BraTS2023/{imagesTr, labelsTr, dataset.json}

Channel order written for nnU-Net:
    _0000 = T1 (t1n)   _0001 = T1ce (t1c)   _0002 = T2 (t2w)   _0003 = FLAIR (t2f)

Region-based training (matches the official BraTS evaluation regions):
    whole_tumor     = labels {1, 2, 3}
    tumor_core      = labels {1, 3}
    enhancing_tumor = label  {3}

Some BraTS releases encode the enhancing tumor as label 4 instead of 3.
This script auto-detects label 4 and remaps 4 -> 3 so labels are contiguous,
which keeps the script correct for both encodings.

Usage:
    python scripts/01_convert_dataset.py --brats_dir /workspace/raw_brats2023/train
"""
import argparse
import os
import shutil
from pathlib import Path

import nibabel as nib
import numpy as np
from nnunetv2.dataset_conversion.generate_dataset_json import generate_dataset_json

# modality suffix -> nnU-Net channel index
MODALITY_TO_CHANNEL = {
    "t1n": "0000",  # T1
    "t1c": "0001",  # T1ce
    "t2w": "0002",  # T2
    "t2f": "0003",  # FLAIR
}


def relabel_and_save(seg_path: Path, out_path: Path) -> None:
    img = nib.load(str(seg_path))
    data = np.asanyarray(img.dataobj)
    if 4 in np.unique(data):          # older BraTS encoding -> make contiguous
        data = data.copy()
        data[data == 4] = 3
    data = np.rint(data).astype(np.uint8)
    nib.save(nib.Nifti1Image(data, img.affine, img.header), str(out_path))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--brats_dir", required=True,
                    help="folder containing the per-case BraTS subfolders")
    ap.add_argument("--dataset_id", type=int, default=137)
    ap.add_argument("--dataset_name", default="BraTS2023")
    ap.add_argument("--nnunet_raw", default=os.environ.get("nnUNet_raw"))
    args = ap.parse_args()

    assert args.nnunet_raw, "Set the nnUNet_raw env var (source env.sh) or pass --nnunet_raw"

    ds_folder = Path(args.nnunet_raw) / f"Dataset{args.dataset_id:03d}_{args.dataset_name}"
    images_tr = ds_folder / "imagesTr"
    labels_tr = ds_folder / "labelsTr"
    images_tr.mkdir(parents=True, exist_ok=True)
    labels_tr.mkdir(parents=True, exist_ok=True)

    case_dirs = sorted(d for d in Path(args.brats_dir).iterdir() if d.is_dir())
    print(f"Found {len(case_dirs)} candidate case folders in {args.brats_dir}")

    n = 0
    for d in case_dirs:
        cid = d.name  # e.g. BraTS-GLI-00000-000
        seg = d / f"{cid}-seg.nii.gz"
        mods = {m: d / f"{cid}-{m}.nii.gz" for m in MODALITY_TO_CHANNEL}
        if not seg.exists() or not all(p.exists() for p in mods.values()):
            print(f"  ! skipping {cid} (missing modality or seg)")
            continue
        for m, ch in MODALITY_TO_CHANNEL.items():
            shutil.copy(mods[m], images_tr / f"{cid}_{ch}.nii.gz")
        relabel_and_save(seg, labels_tr / f"{cid}.nii.gz")
        n += 1
        if n % 100 == 0:
            print(f"  ...converted {n} cases")

    print(f"Converted {n} cases into {ds_folder}")

    generate_dataset_json(
        output_folder=str(ds_folder),
        channel_names={0: "T1", 1: "T1ce", 2: "T2", 3: "FLAIR"},
        labels={
            "background": 0,
            "whole_tumor": (1, 2, 3),
            "tumor_core": (1, 3),
            "enhancing_tumor": (3,),
        },
        regions_class_order=(1, 2, 3),   # WT -> 1, TC -> 2, ET -> 3 in the prediction
        num_training_cases=n,
        file_ending=".nii.gz",
        dataset_name=args.dataset_name,
        description="BraTS 2023 Adult Glioma, region-based training (WT / TC / ET).",
    )
    print("Wrote dataset.json. Done.")


if __name__ == "__main__":
    main()
