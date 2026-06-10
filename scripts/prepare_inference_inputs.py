#!/usr/bin/env python
"""Prepare BraTS case folders for nnU-Net inference.

Copies the 4 modalities of each case into one flat folder using nnU-Net's
channel naming (CASEID_0000..CASEID_0003.nii.gz). No segmentation or
dataset.json is needed for inference - only the images.

    _0000 = T1 (t1n)   _0001 = T1ce (t1c)   _0002 = T2 (t2w)   _0003 = FLAIR (t2f)
"""
import argparse
import shutil
from pathlib import Path

MODALITY_TO_CHANNEL = {"t1n": "0000", "t1c": "0001", "t2w": "0002", "t2f": "0003"}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--brats_dir", required=True, help="folder of BraTS case subfolders")
    ap.add_argument("--out_dir", required=True, help="flat output folder for nnU-Net inputs")
    args = ap.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    cases = sorted(d for d in Path(args.brats_dir).iterdir() if d.is_dir())
    n = 0
    for d in cases:
        cid = d.name
        mods = {m: d / f"{cid}-{m}.nii.gz" for m in MODALITY_TO_CHANNEL}
        if not all(p.exists() for p in mods.values()):
            print(f"  ! skipping {cid} (missing modality)")
            continue
        for m, ch in MODALITY_TO_CHANNEL.items():
            shutil.copy(mods[m], out / f"{cid}_{ch}.nii.gz")
        n += 1
    print(f"Prepared {n} cases -> {out}")


if __name__ == "__main__":
    main()
