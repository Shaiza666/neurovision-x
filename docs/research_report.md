# Reliable 3D glioma segmentation on multimodal MRI: a reproducible methods study

**NeuroVision-X**  
Software version 0.9.0 · repository [https://github.com/Shaiza666/neurovision-x](https://github.com/Shaiza666/neurovision-x)

**This software is an experimental research prototype. It is not a clinically validated diagnostic system. Do not use these outputs for diagnosis or treatment decisions.**

**Empirical status (this repository):** no official BraTS volumes are present. No non-smoke `metrics_test.json` exists. BraTS Dice, HD95, ECE, and ablation tables are **not measured**. One synthetic smoke job was run for software verification only. This document is **not** a published paper and claims no conference or journal acceptance.

---

## Abstract

Automatic glioma sub-region segmentation on co-registered T1, T1ce, T2, and FLAIR MRI is a standard 3D computer-vision task, but overlap-centric training can leave large surface errors on enhancing tumor, and softmax confidence is often poorly calibrated at the patient level. NeuroVision-X is a **reproducible research stack** that implements a matched protocol for 3D U-Net and Attention U-Net, optional Squeeze-and-Excitation fusion, a Kervadec boundary term on enhancing tumor, Monte Carlo dropout, nested WT/TC/ET metrics, calibration scores, and Grad-CAM overlays. All modelling components are **established methods**. The intended contribution is the joint protocol and the empirical evidence that protocol would produce on licensed BraTS data.

That evidence is **not yet available**. Dataset scripts and `scripts/train.py` exit with `MissingBraTSDataError` until official patient folders exist under `data/raw/brats/`. The only completed training job is a two-epoch smoke run on random tensors (CPU, CUDA unavailable). Those losses and Dice values are excluded from research tables. Hypotheses H1–H3 remain open. H3 (CNN vs transformer) is deferred for lack of a matched GPU comparison.

---

## 1. Introduction

Gliomas are spatially heterogeneous. BraTS-style labels distinguish necrotic / non-enhancing core (NCR), peritumoral edema (ED), and enhancing tumor (ET). Evaluation uses nested regions: whole tumor (WT) = NCR ∪ ED ∪ ET, tumor core (TC) = NCR ∪ ET, and ET [4]. Manual contours disagree across raters, especially on ET and infiltrative edema [4]. Challenge volumes are skull-stripped and co-registered; they are not a prospective clinical trial dataset [4], [5].

3D encoder–decoders with skip connections remain the default medical segmentation backbone [1], [2]. Strong pipelines such as nnU-Net show that preprocessing and validation often dominate “novel” blocks [12], [22]. Transformer glioma models exist [14], [16], but unmatched comparisons are a documented source of inflated claims [22].

Two reliability problems motivate this project. First, Dice integrates over regions; thin ET errors barely change overlap while dominating Hausdorff distance [17], [18]. Second, voxel-wise uncertainty maps are weak as local error locators on brain tumors, whereas **patient-aggregated** scores are more useful for detecting failed cases [19], [20].

NeuroVision-X therefore asks an applied question:

> On official BraTS multimodal MRI, does adding a boundary-aware loss to a **matched** 3D U-Net reduce ET Hausdorff error, and can inexpensive test-time dropout identify patients whose segmentations failed, without claiming clinical validation?

This report describes the software that can answer that question, the experiments that have **not** been run, and the single smoke job that has.

---

## 2. Related work

A full catalogue with DOIs is in [literature_review.md](literature_review.md) and [references.md](references.md). Summary:

**Backbones.** U-Net [1] and 3D U-Net [2] define skip-connected volumetric segmentation. V-Net popularized a Dice objective for imbalance [3]. DeepMedic [6] and Myronenko’s BraTS 2018 encoder–decoder with a VAE branch [7] are strong historical glioma CNNs. nnU-Net [12] and nnU-Net Revisited [22] argue that configuration and matched validation beat many specialized 3D networks. MedNeXt [21] is a modern ConvNet alternative to transformers.

**Attention and transformers.** Attention gates on skips (Oktay et al.) were demonstrated on abdominal CT, not glioma [8]. TransUNet is 2D [13]. TransBTS [14], UNETR [15], and Swin UNETR [16] are 3D (the last two with public code; Swin UNETR targets brain-tumor MRI). This project does **not** implement UNETR/Swin UNETR; H3 is deferred.

**Uncertainty.** MC dropout as approximate Bayesian inference [9]; aleatoric vs epistemic decomposition [10]; deep ensembles [11]. On brain tumors, Jungo et al. [19] found dataset-level calibration can look acceptable while **subject-level** calibration is poor. Mehrtash et al. [20] require ECE and reliability diagrams, not Dice alone.

**Boundary losses.** Kervadec et al. [17] integrate softmax against a signed distance map (ISLES/WMH, not BraTS ET as the primary setting). Karimi and Salcudean [18] approximate Hausdorff distance (prostate/liver/pancreas). Direct glioma ET evidence under a matched 3D U-Net is the gap H1 addresses.

**Attribution.** Grad-CAM [24] localizes gradient-weighted activations. Occlusion-style sensitivity follows Zeiler and Fergus [25]. Saliency maps can fail sanity checks [26]. Overlays in this repo are research QC, not clinical explanations ([explainability.md](explainability.md)).

---

## 3. Research gap and hypotheses

Gaps (from [research_gap.md](research_gap.md); **not** claimed as solved):

1. **Overlap vs surface error on ET.** Need a controlled BraTS study that changes only the loss.
2. **Uncertainty vs reliability.** Few BraTS papers report ECE and a patient-level failure detector.
3. **Confounded CNN vs transformer comparisons.** Resource-limited work should not start by inventing a transformer.
4. **Challenge Dice ≠ clinical validation.** Non-clinical framing and leakage-free splits are mandatory.

**H1.** Same 3D U-Net, Dice+CE+boundary on ET vs Dice+CE: lower mean ET HD95, ET Dice not worse than a 1-point non-inferiority margin.

**H2.** Patient mean MC-dropout entropy correlates with patient Dice (Spearman ρ < 0) and detects the worst ET Dice quartile with higher AUPRC than treating voxel entropy as a local error map.

**H3.** Matched 3D U-Net vs UNETR/Swin UNETR within 1 Dice point at similar compute — **deferred** (`unetr` / `swinunetr` raise `NotImplementedError`; this machine recorded `cuda: false`).

**Status:** hypotheses only. They are not results.

---

## 4. Methodology

Implemented pipeline (details: [architecture.md](architecture.md), [proposed_method.md](proposed_method.md)):

1. **Data I/O.** Discover T1/T1ce/T2/FLAIR (and optional seg) from the tree on disk (2021 `_t1/_t1ce/_t2/_flair` and 2023+ `-t1n/-t1c/-t2w/-t2f` tokens). Missing modalities are errors; zeros are not synthesized.
2. **Splits.** Patient-level 70/10/20 or official validation folders; seed 2026; no slice mixing (`configs/data.yaml`).
3. **Preprocessing.** RAS orientation, 1 mm target spacing, non-zero z-score, optional foreground crop, training-time random patches.
4. **Backbones.** In-repo 3D U-Net (default) and Attention U-Net; optional MONAI nets when `optional_monai: true`. SE fusion is a **wrapper**, not an edit to `unet3d.py`.
5. **Loss.** Soft Dice (background excluded) + inverse-frequency CE; optional Kervadec boundary on class 3 (ET).
6. **Training engine.** AdamW, gradient clip, optional AMP, `best.pt` / `latest.pt`, val Dice WT/TC/ET. HD95 is computed on full volumes at evaluation time.
7. **Inference.** Direct forward if spatial size equals the patch; otherwise sliding window.
8. **Uncertainty.** MC dropout (`T=10` planned for BraTS; optional T≈5 on small demo volumes). Flip TTA is implemented for `analyze_reliability.py`. Displayed demo labels stay T=1 argmax; T=1 entropy is the softmax of those logits.
9. **Explainability.** 3D Grad-CAM on the last spatial Conv3d; occlusion sensitivity on small volumes; orthogonal MRI/pred/GT/error/entropy.
10. **Reporting.** `scripts/collect_results.py` reads `experiments/*/`. Smoke folders never enter A/B/C tables.

The YOLO detector (`app.py`) is out of scope for glioma metrics.

---

## 5. Mathematical formulation

Let \(x \in \mathbb{R}^{4 \times D \times H \times W}\) be co-registered T1, T1ce, T2, FLAIR. Network \(f_\theta\) produces logits \(z \in \mathbb{R}^{C \times D \times H \times W}\), \(C=4\), \(p=\mathrm{softmax}(z)\). Nested regions are derived after \(\arg\max\), not from extra heads [4].

**Soft Dice** for class \(c\) (background excluded; empty GT classes omitted) [3]:

\[
\mathrm{Dice}_c = \frac{2\langle p_c, g_c\rangle + \varepsilon}{\|p_c\|_1 + \|g_c\|_1 + \varepsilon}.
\]

**Hard Dice** on binary region masks \(P,G\): \(2|P\cap G|/(|P|+|G|)\). Empty vs empty scores 1; empty vs non-empty scores 0.

**HD95.** 95th percentile of bidirectional surface distances. Empty surface → **NaN** (omitted from means, never replaced by 0) [4], [18]. After 1 mm resampling, one voxel ≈ 1 mm; otherwise do not label HD95 as millimetres.

**Baseline loss.** \(\mathcal{L}_{\mathrm{base}} = \lambda_{\mathrm{dice}}\mathcal{L}_{\mathrm{dice}} + \lambda_{\mathrm{ce}}\mathcal{L}_{\mathrm{ce}}\).

**Boundary loss** [17], \(\phi_G\) signed distance of a ground-truth region (negative inside):

\[
\mathcal{L}_{\mathrm{bd}} = \frac{1}{|\Omega|}\sum_{x\in\Omega} p(x)\,\phi_G(x),\qquad
\mathcal{L} = \mathcal{L}_{\mathrm{base}} + \lambda_{\mathrm{bd}}\mathcal{L}_{\mathrm{bd}}.
\]

H1 uses \(\lambda_{\mathrm{bd}}=0.1\) on ET only (`boundary_classes: [3]`), subject to val-only tuning after approval.

**SE stem** [23] on the four input channels: \(s=\sigma(W_2\,\delta(W_1\,\mathrm{GAP}(x)))\), \(\tilde{x}=s\odot x\).

**Attention gates** [8]: \(\alpha=\sigma(\psi^\top(\delta(W_x x + W_g g)))\), \(x'=\alpha\odot x\).

**MC dropout** [9]: \(\bar{p}\approx\frac{1}{T}\sum_{t=1}^{T}\mathrm{softmax}(f_{\theta,t}(x))\). Predictive entropy \(H(\bar{p})=-\sum_c \bar{p}_c\log\bar{p}_c\). Patient score: mean entropy on WT-candidate voxels [19].

**ECE** bins \(\max_c p_c\) against correctness [20]. **Brier** is mean squared error between softmax and one-hot [27].

**Grad-CAM** [24] (3D): weights from spatially pooled gradients of a class score w.r.t. a Conv3d activation, ReLU, trilinear upsample. Default score: mean logit of class \(c\) on voxels already labelled \(c\).

---

## 6. Dataset description

**Intended primary data.** A public BraTS glioma release (2021-style or 2023+ glioma naming) after accepting organizer terms: four MRI sequences, typically 1 mm isotropic, NCR/ED/ET labels (legacy ET `4` remapped to `3`). Obtain from the official BraTS / Synapse distribution. Do not commit patient files.

**What is in this repository.** `data/raw/brats/` contains only a README. No official NIfTI. Unit-test fixtures in `tests/nifti_fixtures.py` are tiny synthetic arrays, **not** BraTS.

**Loader behaviour.** `src/neurovisionx/data/` inspects the tree that is actually present (patient folders, official train/val subfolders, or flat NIfTI groups). Scripts `validate_dataset.py`, `prepare_splits.py`, `dataset_stats.py`, `dataset_qc.py`, `train.py`, and `evaluate.py` raise `MissingBraTSDataError` when no BraTS-like cases exist.

**Splits (configured, not yet written for real patients).** Patient-level; `train_ratio` 0.70, `val_ratio` 0.10, `test_ratio` 0.20; `seed` 2026; `forbid_slice_mix: true`; honor official validation folders when present.

---

## 7. Experimental protocol

Full tables: [experimental_protocol.md](experimental_protocol.md), [evaluation_protocol.md](evaluation_protocol.md), `configs/ablations.yaml`.

**Status: awaiting approval.** `expensive_training_requires_approval: true`. Do not start BraTS jobs until the protocol is approved **and** licensed volumes exist.

| ID | Train? | Architecture | Fusion | Boundary | Test \(T\) |
| --- | --- | --- | --- | --- | --- |
| A0 | yes | 3D U-Net | early | 0 | 1 |
| A1 | yes | 3D U-Net | SE | 0 | 1 |
| A2 | yes | Attention U-Net | early | 0 | 1 |
| A3 | yes | 3D U-Net | early | ET, 0.1 | 1 |
| A4 | no (reuse A0) | 3D U-Net | early | 0 | 10 |
| A5 | yes | 3D U-Net | SE | ET, 0.1 | 10 |

Phase 5 comparison: **A** = A0, **B** = A2, **C** = A5. Component ablations reuse checkpoints (`C_minus_se`, `C_minus_boundary`, `C_minus_mc`). Default seed `{2026}`. Extra seeds `{2027, 2028}` only if approved.

**Metrics (test patients, then mean ± SD and 95% t-interval when \(n\ge 2\)):** Dice and HD95 WT/TC/ET; precision/recall; small-ET quartile; ECE (15 bins), NLL, Brier; Spearman ρ; AUPRC for worst ET Dice quartile; parameters; wall-clock ms for \(T=1\) and \(T=10\); peak CUDA MB if CUDA exists.

---

## 8. Results and discussion

Source: [experimental_results.md](experimental_results.md), generated from on-disk JSON (`2026-11-17T12:22:45Z`). Missing cells stay **not measured**.

### 8.1 Official BraTS comparison

Research-eligible test evaluations (non-smoke `metrics_test.json`): **0**.

| Metric | A: 3D U-Net | B: Attention U-Net | C: SE + ET boundary + MC |
| --- | --- | --- | --- |
| Dice WT / TC / ET | not measured | not measured | not measured |
| HD95 WT / TC / ET | not measured | not measured | not measured |
| Precision / recall ET | not measured | not measured | not measured |
| ECE, Brier | not measured | not measured | not measured |
| Small-ET Dice | not measured | not measured | not measured |
| Spearman ρ, AUPRC | not measured | not measured | not measured |
| Latency, GPU MB | not measured | not measured | not measured |

**Discussion.** There is no empirical basis to say the proposed combination is more accurate, better calibrated, or faster than the baseline. Any such sentence would be fabrication.

### 8.2 Software verification (not research)

| Item | Recorded value |
| --- | --- |
| Folder | `experiments/smoke/` |
| Status | `smoke_synthetic_not_research` |
| Device | cpu · CUDA false · torch 2.15.0+cpu |
| Epochs | 2 |
| Architecture at train time | 3D U-Net, features `[8, 16]`, patch \(16^3\) |
| Last synthetic val Dice WT / TC / ET | 0.0345 / 0.0241 / 0.0010 |
| Demo load of `best.pt` | 85,542 parameters; 388.6 ms CPU; argmax class counts on one \(16^3\) cube `{0:1218, 1:90, 2:802, 3:1986}` |

These numbers used **random tensors**. They must not be cited as BraTS performance or as support for H1/H2.

### 8.3 Unit tests

`python -m unittest` collected **105** tests (1 skip if an optional import is absent). Tests cover splits, missing-data errors, shapes, losses, smoke checkpoint loading, and demo validation. Passing tests are not Dice on patients.

---

## 9. Ablation study

**Not started** on official data.

| Arm | Removes | Train? | dice_et |
| --- | --- | --- | --- |
| C_minus_se | SE fusion | no (reuse ckpt) | not measured |
| C_minus_boundary | ET boundary | no (reuse ckpt) | not measured |
| C_minus_mc | MC dropout | no (reuse ckpt) | not measured |

`scripts/ablation_plan.py` prints the plan and **does not train**.

---

## 10. Limitations

1. **No licensed BraTS tree** in this workspace; patient metrics do not exist.
2. **H1 and H2 untested.** H3 deferred (no matched transformer GPU run).
3. **Expensive training gated** by `expensive_training_requires_approval: true`.
4. **Smoke Dice is not comparable** to challenge scores.
5. **HD95 NaNs** when a surface cannot be formed; they are omitted, not imputed.
6. **Grad-CAM** is interpolated, layer- and score-dependent, and can trace the prediction itself [24], [26].
7. **Entropy is not error** [19]. T=1 maps in the demo are not MC dropout.
8. **CPU-only recorded environment** (`cuda: false`); GPU memory is undefined here.
9. **Not a clinical device.** Challenge preprocessing is not bedside MRI. Inter-rater disagreement is large [4].
10. **YOLO demo** is 2D natural-image detection and is not a glioma result.

Unresolved: BraTS year/release actually obtained; GPU approval for H3; whether A0 can finish under available compute.

---

## 11. Conclusion and future work

NeuroVision-X is a documented, tested **methods codebase** for a reliability study of 3D glioma segmentation. It does not yet contain the study’s measurements. The correct conclusion is therefore limited:

- The literature gaps and hypotheses are stated and cited.
- The software implements matched backbones, losses, uncertainty, metrics, and a checkpoint-backed demo.
- Official BraTS A/B/C results, ablations, and hypothesis tests are **incomplete**.

**Future work (in order):** (i) accept a BraTS license and unpack volumes; (ii) freeze `data/splits/patients.json`; (iii) approve the protocol; (iv) train A0; (v) evaluate and run A4 (H2) on A0; (vi) train A3 (H1) if compute remains; (vii) A1/A2/A5 only after that; (viii) H3 only if a transformer fits under the same protocol. Negative H1/H2 outcomes will be reported if those jobs run.

---

## 12. References

IEEE-style list: [references.md](references.md). BibTeX: [references.bib](references.bib). Identifiers were checked against publisher, PubMed, PMLR, NeurIPS, CVF, or arXiv pages. Conference papers without a Crossref DOI are given an official URL instead of a fabricated DOI.
