#!/usr/bin/env python
"""Render a prediction-vs-ground-truth overlay PNG for a BraTS case.

Produces a clean multi-panel figure (MRI background | ground truth | prediction)
on the most informative axial slice, with the three BraTS regions colour-coded
and per-region 3D Dice printed in the title. Designed as the hero visual for a
blog post.

Key correctness detail (why we don't just colour by raw label):
  * The GROUND TRUTH is a sub-region label map. By default we assume the
    encoding produced by scripts/01_convert_dataset.py: 1=NCR, 2=ED, 3=ET, so
    tumor core = {1,3} and enhancing = {3}. Override with --gt-tc-labels /
    --gt-et-labels if your conversion differs.
  * The PREDICTION from a region-based nnU-Net is ORDINAL-NESTED because
    regions_class_order=(1,2,3): value >=1 is whole tumor, >=2 is tumor core,
    >=3 is enhancing tumor. We derive regions that way - robust regardless of
    the sub-region naming.
Both are mapped to the same 3-tier scheme (1=WT-only/edema, 2=core, 3=ET) so the
GT and prediction panels are directly comparable with one colour map.

Usage:
  python scripts/overlay_prediction.py \
      --image  /workspace/preds/_nnunet_inputs/BraTS-GLI-00000-000_0003.nii.gz \
      --pred   /workspace/preds/predictions/BraTS-GLI-00000-000.nii.gz \
      --gt     /workspace/nnUNet_raw/Dataset137_BraTS2023/labelsTr/BraTS-GLI-00000-000.nii.gz \
      --out    overlay_BraTS-GLI-00000-000.png
"""
import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import nibabel as nib
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch

# tier colours: 1 = WT-only (edema), 2 = tumor core, 3 = enhancing tumor
TIER_COLORS = ["#FFD54F", "#4FC3F7", "#E53935"]
TIER_LABELS = ["Whole tumor (edema rim)", "Tumor core", "Enhancing tumor"]


def load(path):
    img = nib.load(str(path))
    return np.asanyarray(img.dataobj)


def gt_tiers(seg, tc_labels, et_labels):
    """GT sub-region map -> nested tier map (1=WT-only, 2=core, 3=ET)."""
    out = np.zeros(seg.shape, dtype=np.uint8)
    out[seg > 0] = 1
    out[np.isin(seg, tc_labels)] = 2
    out[np.isin(seg, et_labels)] = 3
    return out


def pred_tiers(seg):
    """Region-ordinal prediction -> nested tier map (1/2/3)."""
    out = np.zeros(seg.shape, dtype=np.uint8)
    out[seg >= 1] = 1
    out[seg >= 2] = 2
    out[seg >= 3] = 3
    return out


def dice(a, b):
    a, b = a.astype(bool), b.astype(bool)
    denom = a.sum() + b.sum()
    if denom == 0:
        return 1.0  # both empty -> nothing to find, perfect agreement
    return 2.0 * np.logical_and(a, b).sum() / denom


def best_slice(mask3d, axis=2):
    """Axial slice index with the largest foreground area."""
    areas = mask3d.sum(axis=tuple(i for i in range(3) if i != axis))
    return int(np.argmax(areas)) if areas.max() > 0 else mask3d.shape[axis] // 2


def take_slice(vol, z, axis=2):
    sl = [slice(None)] * 3
    sl[axis] = z
    return np.rot90(vol[tuple(sl)])  # rot90 for a conventional upright view


def norm_img(slice2d):
    lo, hi = np.percentile(slice2d, [1, 99])
    if hi <= lo:
        hi = lo + 1
    return np.clip((slice2d - lo) / (hi - lo), 0, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True, help="background MRI (e.g. FLAIR _0003)")
    ap.add_argument("--pred", required=True, help="predicted segmentation .nii.gz")
    ap.add_argument("--gt", default=None, help="ground-truth segmentation .nii.gz (optional)")
    ap.add_argument("--out", default="overlay.png")
    ap.add_argument("--slice", type=int, default=None, help="axial slice index (default: auto)")
    ap.add_argument("--axis", type=int, default=2, help="slice axis (0/1/2, default 2 = axial)")
    ap.add_argument("--alpha", type=float, default=0.45)
    ap.add_argument("--modality-name", default="FLAIR")
    ap.add_argument("--gt-tc-labels", default="1,3", help="GT raw labels that form tumor core")
    ap.add_argument("--gt-et-labels", default="3", help="GT raw labels that form enhancing tumor")
    args = ap.parse_args()

    mri = load(args.image).astype(np.float32)
    pred = load(args.pred)
    pt = pred_tiers(pred)

    has_gt = args.gt is not None
    if has_gt:
        gt = load(args.gt)
        tc_labels = [int(x) for x in args.gt_tc_labels.split(",") if x != ""]
        et_labels = [int(x) for x in args.gt_et_labels.split(",") if x != ""]
        gtt = gt_tiers(gt, tc_labels, et_labels)
    else:
        gtt = None

    # pick the slice where the most specific region is largest, so the hero
    # panel shows all three tiers (ET is nested inside TC inside WT)
    if args.slice is not None:
        z = args.slice
    else:
        ref = gtt if has_gt else pt
        for tier in (3, 2, 1):
            if (ref >= tier).any():
                z = best_slice(ref >= tier, args.axis)
                break
        else:
            z = ref.shape[args.axis] // 2

    mri_s = norm_img(take_slice(mri, z, args.axis))
    pt_s = take_slice(pt, z, args.axis)
    gtt_s = take_slice(gtt, z, args.axis) if has_gt else None

    cmap = ListedColormap(TIER_COLORS)

    def draw(ax, title, tier_slice=None):
        ax.imshow(mri_s, cmap="gray", interpolation="nearest")
        if tier_slice is not None:
            masked = np.ma.masked_where(tier_slice == 0, tier_slice)
            ax.imshow(masked, cmap=cmap, vmin=1, vmax=3, alpha=args.alpha,
                      interpolation="nearest")
        ax.set_title(title, fontsize=12)
        ax.axis("off")

    n_panels = 3 if has_gt else 2
    fig, axes = plt.subplots(1, n_panels, figsize=(4.2 * n_panels, 4.8))

    draw(axes[0], f"{args.modality_name} (axial slice {z})")
    if has_gt:
        draw(axes[1], "Ground truth", gtt_s)
        draw(axes[2], "Prediction", pt_s)
        d_wt = dice(gtt >= 1, pt >= 1)
        d_tc = dice(gtt >= 2, pt >= 2)
        d_et = dice(gtt == 3, pt == 3)
        suptitle = f"Dice  WT {d_wt:.3f}   TC {d_tc:.3f}   ET {d_et:.3f}"
    else:
        draw(axes[1], "Prediction", pt_s)
        suptitle = "Prediction (no ground truth provided)"

    legend = [Patch(facecolor=c, edgecolor="none", label=l)
              for c, l in zip(TIER_COLORS, TIER_LABELS)]
    fig.legend(handles=legend, loc="lower center", ncol=3, frameon=False,
               bbox_to_anchor=(0.5, -0.02), fontsize=10)
    fig.suptitle(suptitle, fontsize=13, y=1.02)
    fig.tight_layout()
    fig.savefig(args.out, dpi=200, bbox_inches="tight", facecolor="white")
    print(f"Wrote {args.out}  (slice {z})")
    if has_gt:
        print(f"  Dice  WT={d_wt:.4f}  TC={d_tc:.4f}  ET={d_et:.4f}")


if __name__ == "__main__":
    main()
