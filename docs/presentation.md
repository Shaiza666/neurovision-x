---
marp: true
theme: default
paginate: true
title: NeuroVision-X research slides
---

<!--
Research presentation (Phase 8). Export with Marp if a PDF is needed:
https://github.com/marp-team/marp
This file is also readable as GitHub markdown.
-->

# NeuroVision-X

A reproducible 3D glioma MRI segmentation **methods stack**

**Experimental research prototype · not a clinically validated diagnostic system**

Repository: [github.com/Shaiza666/neurovision-x](https://github.com/Shaiza666/neurovision-x)

*Not a published paper. No BraTS training run in this repository.*

---

# Problem

- Nested glioma regions: **WT / TC / ET** on T1, T1ce, T2, FLAIR [Menze 2015]
- Dice can look strong while **ET Hausdorff** fails
- Softmax confidence is often **miscalibrated per patient** [Jungo 2020]
- Architecture “wins” are often **confounded by preprocessing** [Isensee 2024]

---

# Research question

On **official BraTS** data, with a **matched** 3D U-Net:

1. Does a **boundary term on ET** reduce ET HD95 without a material Dice drop? **(H1)**
2. Does **patient-level MC-dropout entropy** flag failed cases better than voxel maps? **(H2)**

**H3** (CNN vs transformer, matched compute) is **deferred** — no GPU transformer run here.

---

# Established methods (not invented here)

| Component | Source |
| --- | --- |
| 3D U-Net | Çiçek et al., 2016 |
| Attention U-Net | Oktay et al., 2018 |
| SE fusion | Hu et al., 2018 |
| Dice + CE | Milletari et al., 2016 |
| Boundary loss | Kervadec et al., 2021 |
| MC dropout | Gal & Ghahramani, 2016 |
| Grad-CAM | Selvaraju et al., 2017 |

**Contribution = joint protocol + software + (future) matched evidence.**

---

# System (implemented)

```
NIfTI → validate / patient splits / z-score
      → 3D U-Net or Attention U-Net (± SE)
      → Dice+CE (± ET boundary)
      → checkpoints
      → Dice/HD95/ECE · MC dropout · Grad-CAM
      → Streamlit research dashboard
```

YOLOv8n `app.py` is a **separate** 2D detector, preserved, not MRI.

---

# Dataset in this repository

- Official BraTS NIfTI: **absent** (`data/raw/brats/` is a README only)
- `train.py` exits with `MissingBraTSDataError`
- No patient-level split file for real cases
- Test fixtures are **tiny synthetic arrays**, not challenge data

Obtain volumes from the official BraTS / Synapse distribution after accepting their license.

---

# Protocol (awaiting approval)

| Arm | Isolates |
| --- | --- |
| A0 | 3D U-Net baseline |
| A1 | SE fusion only |
| A2 | Attention U-Net |
| A3 | ET boundary (H1) |
| A4 | MC dropout on A0 (H2) — no extra train |
| A5 | Combination (not claimed SOTA) |

`expensive_training_requires_approval: true`

---

# Results on official BraTS

# **Not measured**

MSD Task01 reduced CPU results are in `docs/experimental_results.md`. They are not this slide.

| Dice / HD95 / ECE | A | B | C |
| --- | --- | --- | --- |
| WT, TC, ET | — | — | — |

Research-eligible `metrics_test.json` files: **0**

No improvement of the proposed method is claimed.

---

# What actually ran

**Smoke job** (`status: smoke_synthetic_not_research`)

- 2 epochs · CPU · CUDA false · torch 2.15.0+cpu
- Random tensors, patch \(16^3\), features `[8, 16]`
- Synthetic val Dice WT/TC/ET ≈ **0.034 / 0.024 / 0.001**
- **Do not cite as BraTS**

Dashboard load of `best.pt`: **85,542** params, **388.6 ms** CPU (one cube)

---

# Software delivered

- Config-driven PyTorch / MONAI research package (`src/neurovisionx/`)
- Leakage-aware dataset layer · honest missing-data errors
- Evaluation + reliability scripts that refuse smoke as BraTS
- 3D Grad-CAM / overlays with written limits
- Streamlit demo: **real checkpoint only**, no mock logits
- **105** unit tests collected

---

# Limitations (unresolved)

- No licensed MRI on disk
- H1 / H2 untested; H3 deferred
- Smoke Dice ≠ challenge Dice
- Grad-CAM ≠ causal explanation
- Entropy ≠ error
- Not a clinical device

---

# Conclusion

**Built:** a reproducible reliability stack for 3D glioma segmentation.

**Not built yet:** the BraTS numbers that would accept or reject H1 and H2.

Next: licensed data → freeze splits → approve protocol → train **A0**.

---

# Not for clinical use

NeuroVision-X is an experimental research prototype.

It is **not** a clinically validated diagnostic system.

Do not use these outputs for diagnosis or treatment decisions.
