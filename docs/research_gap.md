# Research gaps

**Project:** NeuroVision-X, Phase 1  
**Rule:** every gap below is tied to a cited paper in `literature_review.md`. Gaps are **open problems in the literature**, not claims that NeuroVision-X has already solved them.

This software is not a clinically validated diagnostic tool.

## Established consensus (not a gap)

The community already knows that:

- 3D encoder–decoders with skip connections work for volumetric MRI (Çiçek et al., 2016; Milletari et al., 2016).
- Dice + cross-entropy is the default regional objective (Milletari et al., 2016; Myronenko, 2019).
- nnU-Net-style configuration is a very strong baseline and often beats “novel” 3D networks under matched validation (Isensee et al., 2021, 2024).
- BraTS nested regions (WT / TC / ET) and HD95 are the reporting standard (Menze et al., 2015).
- MC dropout, aleatoric/epistemic decomposition, and deep ensembles exist (Gal and Ghahramani, 2016; Kendall and Gal, 2017; Lakshminarayanan et al., 2017).
- Boundary and Hausdorff losses exist for unbalanced organs and lesions (Kervadec et al., 2021; Karimi and Salcudean, 2020).

A Master’s project that only “proposes another 3D U-Net” or “adds a Transformer because it is new” is **not** a contribution. Isensee et al. (2024) document that pattern as a validation failure, not as progress.

## Gap 1 — Overlap-trained models remain brittle at tumor boundaries

**Evidence.** Dice integrates over regions. Karimi and Salcudean (2020) show that networks trained this way can keep Dice while HD stays large. Kervadec et al. (2021) motivate a contour-based loss for rare, thin structures. BraTS ET is a small, high-stakes class; WT Dice can look strong while ET HD95 fails.

**What is missing.** A **controlled BraTS study** that keeps the same 3D backbone, preprocessing, and split, and only changes the loss (Dice+CE vs Dice+CE+boundary or HD approximation), reporting **ET HD95 and ET Dice together**. Kervadec’s main empirical setting is ISLES/WMH; Karimi’s is prostate/liver/pancreas. Direct glioma evidence is thinner than the method papers suggest.

## Gap 2 — Uncertainty maps are not reliable enough to trust per voxel

**Evidence.** Jungo et al. (2020) evaluate uncertainty **on brain tumor segmentation** and find decent dataset-level calibration, **miscalibration at the subject level**, and limited usefulness of voxel-wise maps for correcting local errors. Aggregated (patient-level) uncertainty is more useful for detecting failed segmentations. Mehrtash et al. (2020) show that segmentation softmaxes are often overconfident unless calibration is measured (ECE, reliability diagrams).

**What is missing.** Most BraTS papers still publish only Dice/HD95. Few report **ECE / NLL** and a **patient-level failure detector** (does high mean entropy predict low Dice?). That is a reliability gap, not an architecture gap.

## Gap 3 — Transformer vs CNN comparisons are often confounded

**Evidence.** TransBTS (Wang et al., 2021) and Swin UNETR (Hatamizadeh et al., 2022) are real 3D glioma transformer papers. TransUNet (Chen et al., 2021) is 2D and not glioma. Isensee et al. (2024) argue that many 3D “wins” disappear when nnU-Net-quality preprocessing and validation are used. MedNeXt (Roy et al., 2023) shows modern ConvNets can match transformer-style scaling.

**What is missing.** For a resource-limited Master’s project, the honest question is not “invent a new Transformer.” It is: **under identical MONAI pipelines, does a 3D CNN baseline already match a transformer of similar compute on BraTS?** If yes, compute should go to loss design and uncertainty, not another encoder.

## Gap 4 — Challenge performance is not clinical reliability

**Evidence.** Menze et al. (2015) already report substantial **inter-rater** disagreement. Bakas et al. (2017) and later BraTS releases use heavily preprocessed, skull-stripped, co-registered volumes. None of the method papers above is a prospective clinical validation.

**What is missing.** Explicit **non-clinical** framing, patient-level splits with no leakage, and reporting of failures. NeuroVision-X must not describe itself as diagnostic software.

## What NeuroVision-X will not claim

- A new backbone that “beats nnU-Net.”
- Clinical accuracy, FDA/CE marking, or diagnostic use.
- Voxel-wise uncertainty as a proven error locator (Jungo et al. argue against naive use).
- Results on synthetic “MRI” tensors as if they were BraTS.

## Feasible gap to attack in this project

Combine **Gap 1** and **Gap 2** on a **single, well-controlled 3D CNN** (Gap 3 says not to start with an unproven transformer):

> Train a 3D U-Net / DynUNet-style model on official BraTS data with patient-level splits; add a boundary-aware term aimed at ET/TC surfaces; estimate MC-dropout or TTA uncertainty; report Dice, HD95, and calibration, including a patient-level failure analysis.

That is an **applied, reproducible reliability study**. It uses established methods. The contribution is the **joint protocol and the empirical evidence**, not a new named architecture.
