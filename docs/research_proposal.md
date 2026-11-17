# Preliminary research proposal

**Title:** Reliable 3D glioma segmentation on multimodal MRI: a controlled study of boundary-aware training and uncertainty  
**Project:** NeuroVision-X  
**Programme framing:** Erasmus Mundus Master’s portfolio (reproducible methods study, not a clinical device)  
**Phase:** 1 — plan only (historical). Software Phases 2–7 were implemented later; **one real BraTS training run has still not been completed.** Do not read this file as evidence that Stage 3 finished.

This software is not a clinically validated diagnostic tool. No segmentation model has been trained in this repository for BraTS.

## 1. Motivation

Manual glioma delineation on T1, T1-Gd, T2, and FLAIR is slow and disagrees across raters (Menze et al., 2015). Automatic 3D CNNs and transformers now post high BraTS Dice, but (i) surface errors on enhancing tumor remain, (ii) softmax confidence is poorly calibrated at the patient level (Jungo et al., 2020; Mehrtash et al., 2020), and (iii) architecture papers are often confounded by preprocessing (Isensee et al., 2024).

A Master’s-scale contribution that is **honest** is not a new backbone. It is a **reliability protocol**: same data, same split, same 3D U-Net, with and without a boundary term, plus uncertainty scores that are evaluated as failure detectors.

## 2. Research question

> On official BraTS multimodal MRI, does adding a boundary-aware loss to a 3D U-Net reduce ET Hausdorff error, and can inexpensive test-time uncertainty identify patients whose segmentations failed, without claiming clinical validation?

## 3. Established methods vs proposed work

| Established (we use, we do not invent) | Proposed in this project |
| --- | --- |
| 3D U-Net, Dice+CE, MONAI loading/spacing | Joint **BraTS** evaluation of overlap, HD95, and calibration |
| nnU-Net as the literature SOTA reference (Isensee et al., 2021, 2024) | We implement a **MONAI 3D U-Net / DynUNet** baseline we can fully log; full nnU-Net is an optional external comparison, not a claimed original method |
| Boundary / HD losses (Kervadec; Karimi) | Transfer and measure them on **ET/TC**, which those papers did not treat as the primary BraTS setting |
| MC dropout / TTA / ensembles | Patient-aggregated uncertainty as a **failure score** (following Jungo’s negative finding on voxel-wise correction) |
| BraTS nested metrics | Mandatory reporting of unsuccessful runs and leakage-free splits |

## 4. Data

- **Primary:** a public BraTS glioma release (e.g. BraTS 2021 or later) after accepting the challenge data terms. Four modalities, 1 mm isotropic, labels for NCR/ED/ET.  
- **Splits:** patient-level. Suggested 70/10/20 or official train/val if test labels are unavailable (then val is the reporting split and is frozen).  
- **Leakage controls:** no slice mixing; augmentations only on training patients; intensity stats from training only.  
- **Forbidden:** synthetic NIfTI used as reported medical results; using YOLO demo images as MRI.

Data download is **Stage 1**, after this proposal is approved and the BraTS license is accepted.

## 5. Methods (to implement only after approval)

1. **Baseline A:** MONAI 3D U-Net, 4-channel input, 3-class (or 3 nested) output, Dice+CE, deep supervision optional.  
2. **Baseline B (if VRAM allows):** SegResNet / Myronenko-style encoder–decoder.  
3. **Optional C:** UNETR or Swin UNETR from MONAI, **matched** patch size and epochs (tests H3).  
4. **Proposed training variant:** Baseline A + Kervadec-style boundary term on ET (and optionally TC).  
5. **Uncertainty:** MC dropout (\(T \approx 10\)–20) or TTA; report ECE and patient-level mean entropy.  
6. **No silent fallback:** if training fails, the log says it failed. Dummy Dice will not be written.

Mathematical losses are in `research_hypotheses.md`.

## 6. Baselines and metrics

**Must-run baseline:** 3D U-Net + Dice + CE.  
**Literature ceiling (not necessarily re-run in Phase 2):** nnU-Net (Isensee et al., 2021). If we cannot retrain it, we cite published BraTS numbers as **external** and do not pretend they were produced here.

**Segmentation metrics (per patient, then mean ± std):**

- Dice WT, TC, ET  
- HD95 WT, TC, ET (mm)  
- Sensitivity / specificity for ET (optional)

**Reliability metrics (H2):**

- ECE, NLL on voxels (with class handling documented)  
- Spearman correlation of patient uncertainty vs Dice  
- AUPRC for detecting the worst Dice quartile

**Not metrics:** “looks good” overlays without numbers; challenge screenshots without our split.

## 7. Compute and feasibility

A 3D U-Net on \(128^3\) or \(96^3\) patches with mixed precision is the intended first experiment. Full nnU-Net 5-fold and Swin UNETR full-resolution training may exceed a single consumer GPU. The proposal **degrades gracefully**: H1 → H2 → H3.

## 8. Work plan after approval

| Stage | Content | Exit criterion |
| --- | --- | --- |
| 0 (done as YOLO demo) | Preserve existing detector under `app/` when the research tree is added | Existing tests still pass |
| 1 | Configs, MONAI dataset I/O, patient split tests | Split-leakage tests green; **no** fake MRI metrics |
| 2 | 3D U-Net + Dice/CE in `src/` | Shape tests on random tensors only |
| 3 | One real BraTS training run | Log + checkpoint exist; numbers taken from that run |
| 4 | Boundary-loss ablation (H1) | Table of Dice/HD95 |
| 5 | Uncertainty (H2) | ECE + patient-level analysis |
| 6 | Transformer only if H3 is approved and GPU allows | Matched comparison or written deferral |

## 9. Risks

- BraTS access delay.  
- GPU memory: drop transformer, reduce patch size, gradient checkpointing.  
- Boundary loss instability: grid-search \(\lambda_{\mathrm{bd}}\) on val only.  
- Over-claiming: every public sentence must include the non-clinical disclaimer.

## 10. Expected portfolio output

A short methods report with: literature table, three hypotheses, one implemented 3D pipeline, **actual** training logs, and a limitations section. Negative H1/H2 results are acceptable and will be reported.

## 11. Approval gate

**Do not start Stage 1 (data/MONAI) until this plan is approved.** Remaining decisions for the user:

1. Approve Phase 1 documents.  
2. Confirm BraTS data access / which year.  
3. Confirm GPU (CUDA yes/no).  
4. Confirm H3 in or out of scope.
