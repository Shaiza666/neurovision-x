# Abstract

**NeuroVision-X is an experimental research prototype, not a clinically validated diagnostic system.**

Automatic glioma sub-region segmentation on co-registered T1, T1ce, T2, and FLAIR MRI is a standard 3D computer-vision task, but overlap-centric training can leave large surface errors on enhancing tumor, and softmax confidence is often poorly calibrated at the patient level. NeuroVision-X is a **reproducible research stack** that implements a matched protocol for 3D U-Net and Attention U-Net, optional Squeeze-and-Excitation fusion, a Kervadec boundary term on enhancing tumor, Monte Carlo dropout, nested WT/TC/ET metrics, calibration scores, and Grad-CAM overlays. All modelling components are **established methods**. The intended contribution is the joint protocol and the empirical evidence that protocol would produce on licensed BraTS data.

That evidence is **not yet available**. Dataset scripts and `scripts/train.py` exit with `MissingBraTSDataError` until official patient folders exist under `data/raw/brats/`. The only completed training job is a two-epoch smoke run on random tensors (CPU, CUDA unavailable). Those losses and Dice values are excluded from research tables. Hypotheses H1–H3 remain open. H3 (CNN vs transformer) is deferred for lack of a matched GPU comparison.

This document is **not** a published article. No conference or journal acceptance is claimed.

Full write-up: [research_report.md](research_report.md).
