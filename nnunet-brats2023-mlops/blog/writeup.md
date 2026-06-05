# Training nnU-Net for Brain Tumor Segmentation: a 5-Fold, Fully-Tracked MLOps Walkthrough

*A portfolio write-up. Replace the bracketed prompts with your own results and screenshots.*

## TL;DR
One-paragraph summary: what you built (nnU-Net 3d_fullres, region-based, 5-fold CV on
BraTS 2023), where it ran (RunPod RTX 4090), how it was tracked (Weights & Biases),
the headline number ([overall mean Dice] across folds), and the total cost ([$X]).

## Why this project
- The problem: multi-class brain tumor segmentation (WT / TC / ET) on multi-modal MRI.
- Why nnU-Net: self-configuring, strong baseline, the standard to beat in medical imaging.
- The MLOps angle: reproducibility, experiment tracking, and cost-aware compute.

## The data
- BraTS 2023 Adult Glioma: 1,251 cases, 4 modalities (T1, T1ce, T2, FLAIR).
- Region-based labels and why they matter (nested WT / TC / ET regions).
- The label-4 vs label-3 gotcha and how the conversion script handles it.

## Design decisions (the cost/accuracy trade-off)
- 250 epochs instead of the 1000 default: what it saves, what it costs in Dice.
- Why an RTX 4090 is enough (VRAM headroom for standard 3d_fullres).
- Spot instances + checkpoint-resume: how the training survives preemption.
- [Insert your W&B chart of validation Dice vs epoch here.]

## Results
- Per-fold and mean +/- std Dice table (WT / TC / ET). [Embed the W&B summary table.]
- Qualitative example: an overlay of prediction vs ground truth on one validation case.
- Observations: which region was hardest, training stability, where 250 epochs plateaued.

## What I'd do next
- Full 1000-epoch run, ResEnc presets, postprocessing, ensembling, test-set inference.

## Reproduce it
Link to the GitHub repo and the one-line summary of the run order
(setup -> convert -> preprocess -> train -> find-best-config -> aggregate).
