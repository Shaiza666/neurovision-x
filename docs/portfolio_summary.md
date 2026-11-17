# Research portfolio summary

**Software:** NeuroVision-X  
**Repository:** [https://github.com/Shaiza666/neurovision-x](https://github.com/Shaiza666/neurovision-x)  
**Status:** reproducible 3D glioma-segmentation *methods stack* (Phases 1–8). **No official BraTS training run. No paper acceptance, award, or leaderboard entry is claimed.**

This software is an experimental research prototype, not a clinically validated diagnostic system.

## Technical contribution

The contribution is **not** a new backbone and **not** a BraTS score. Isensee et al. (2024) document that unmatched “novel architecture” stories are a common validation failure. This project instead implements a **controlled reliability protocol** that can be run once licensed multimodal MRI is present:

1. **Matched comparison plan.** Same patient-level split, preprocessing YAML, patch size, and seed for 3D U-Net (Çiçek 2016), Attention U-Net (Oktay 2018), optional SE fusion (Hu 2018), ET boundary loss (Kervadec 2021), and MC dropout (Gal and Ghahramani 2016).
2. **Honest measurement contract.** Missing JSON is written as **not measured**. Smoke Dice is excluded from A/B/C tables. Dummy logits are refused.
3. **Joint metrics, when data exist.** Nested WT/TC/ET Dice and HD95, ECE/NLL/Brier, patient-level entropy vs Dice, small-ET stratum — implemented, **not yet filled**.
4. **Research QC, not clinical XAI.** 3D Grad-CAM and entropy overlays with documented limits (Selvaraju 2017; Jungo 2020; Adebayo 2018).

Hypotheses H1 (ET HD95 with boundary loss) and H2 (patient-aggregated uncertainty as a failure score) remain **untested** on glioma MRI. H3 (CNN vs transformer under matched compute) is **deferred** (no GPU comparison in this workspace; `cuda: false`).

## Implementation work

| Phase | Built | Not done |
| --- | --- | --- |
| 1 | Literature review with verified DOIs; gaps; H1–H3 | — |
| 2 | BraTS-style inventory, validation, patient splits, z-score, patches | No official NIfTI on disk |
| 3 | PyTorch 3D U-Net, Attention U-Net, Dice+CE, AMP, checkpoints | No BraTS `best.pt` |
| 4 | SE wrapper, Kervadec boundary, MC dropout, ablation YAML | A0–A5 not trained |
| 5 | Evaluation, reliability, `collect_results.py` | A/B/C cells empty |
| 6 | 3D Grad-CAM, orthogonal overlays | No patient figures |
| 7 | Streamlit dashboard on **real** checkpoints | Patient demo needs BraTS + non-smoke weights |
| 8 | This documentation set | — |

Also preserved: the existing **YOLOv8n ONNX** detector (`app.py`, `neurovision_x/`), which is a 2D natural-image demo and is not used as MRI.

**Recorded execution (this workspace):** one smoke job, 2 epochs, CPU, PyTorch 2.15.0+cpu, CUDA false. Synthetic val Dice WT/TC/ET ≈ 0.034 / 0.024 / 0.001 on **random tensors**. Interface test of `experiments/smoke/best.pt`: 85,542 parameters, 388.6 ms CPU forward on a 16³ cube. Those figures verify code paths. They are **not** BraTS performance.

**Tests:** 105 unit tests collected (`unittest`), 1 skipped when an optional dependency is missing.

## Relevance to advanced computer vision studies

The stack exercises topics that appear in graduate CV and medical-image analysis, with the methods labelled as **established** rather than invented here:

- **Volumetric encoder–decoders** and skip connections (3D U-Net).
- **Class imbalance** (Dice, inverse-frequency CE) vs **surface error** (HD95, boundary loss).
- **Calibration and uncertainty** (MC dropout, ECE, patient-level aggregation).
- **Experimental design** (patient-level splits, leakage controls, pre-registered ablations).
- **Attribution limits** (Grad-CAM as localization, not proof of correctness).
- **Engineering for honesty** (config-driven runs, JSON-only reporting, missing-data errors).

A follow-on study that actually trains A0 then A3 on licensed BraTS would be the empirical test of H1. Until that job finishes, the correct scientific statement is: **the protocol and software exist; the glioma measurements do not.**
