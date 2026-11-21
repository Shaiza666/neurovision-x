# Abstract

**NeuroVision-X is an experimental research prototype, not a clinically validated diagnostic system.**

Automatic glioma sub-region segmentation on co-registered T1, T1ce, T2, and FLAIR MRI is a standard 3D computer-vision task, but overlap-centric training can leave large surface errors on enhancing tumor, and softmax confidence is often poorly calibrated at the patient level. NeuroVision-X is a **reproducible research stack** that implements a matched protocol for 3D U-Net and Attention U-Net, optional Squeeze-and-Excitation fusion, a Kervadec boundary term on enhancing tumor, Monte Carlo dropout, nested WT/TC/ET metrics, calibration scores, and Grad-CAM overlays. All modelling components are **established methods**. The intended contribution is the joint protocol and the empirical evidence that protocol would produce on licensed BraTS data.

Official Synapse BraTS volumes are still absent, and `scripts/train.py` with the default config still exits until `data/raw/brats/` is filled. A separate open path, Medical Segmentation Decathlon Task01 (CC-BY-SA 4.0), was trained on this CPU-only VM under a reduced protocol (2 mm, 96³, widths `[8, 16, 32, 64]`, 25 epochs, 48/8/12 patients, seed 2026). Test Dice ET was 0.464 ± 0.281 (A, 3D U-Net), 0.680 ± 0.257 (B, Attention U-Net), and 0.638 ± 0.253 (C, SE + ET boundary + MC dropout T=10), each n=12. Those cells are not a BraTS leaderboard score. H1 and H2 were not isolated. H3 stays deferred (no GPU).

This document is **not** a published article. No conference or journal acceptance is claimed.

Full write-up: [research_report.md](research_report.md).
