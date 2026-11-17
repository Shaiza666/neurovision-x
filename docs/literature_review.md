# Literature review: reliable 3D brain tumor segmentation from multimodal MRI

**Project:** NeuroVision-X, Phase 1  
**Scope:** glioma sub-region segmentation on co-registered multimodal MRI (T1, T1-Gd/T1ce, T2, FLAIR)  
**Status:** literature only. No models were trained for this document.  
**Disclaimer:** this software is a research prototype. It is not a clinically validated diagnostic tool.

This review uses **verified** publications only. Each entry below includes title, authors, year, venue, and a DOI or official URL checked against publisher, PubMed, PMLR, NeurIPS, CVF, or arXiv pages. Where a conference paper has no Crossref DOI, that is stated and an official proceedings URL is given.

## 1. Problem

Gliomas are spatially heterogeneous. Standard BraTS labels distinguish necrotic/non-enhancing core (NCR), peritumoral edema (ED), and enhancing tumor (ET). Evaluation uses nested regions:

- **Whole tumor (WT)** = NCR ∪ ED ∪ ET  
- **Tumor core (TC)** = NCR ∪ ET  
- **Enhancing tumor (ET)**

The clinical and computational difficulty is not only overlap (Dice). Inter-rater disagreement is large, especially on ET and infiltrative edema. Small false positives on ET inflate Hausdorff distance. Domain shift across scanners is common. A **reliable** system should therefore report overlap, surface error, and **calibrated uncertainty**, not Dice alone.

## 2. Foundational architectures

### 2.1 U-Net and 3D U-Net

Ronneberger, Fischer, and Brox introduced the encoder–decoder with skip connections that remains the default medical segmentation backbone. Çiçek et al. replaced 2D operations with 3D convolutions so the model can use volumetric context. Milletari et al. (V-Net) added residual 3D blocks and a **Dice** objective, which is the standard response to extreme class imbalance (few tumor voxels vs brain).

**Strengths.** Local spatial precision, modest data hunger relative to transformers, well supported in MONAI.  
**Limitations.** Limited long-range context; 3D kernels are memory-heavy; Dice can ignore thin, far-away errors that dominate HD95.

### 2.2 Multi-scale 3D CNNs and BraTS-winning encoder–decoders

Kamnitsas et al. (DeepMedic) used dual-pathway 3D CNNs plus a fully connected CRF and ranked highly on BraTS 2015 / ISLES 2015. Myronenko’s BraTS 2018 winner used an asymmetric encoder–decoder (later related to SegResNet) with a **variational autoencoder** branch to regularize the encoder on a small dataset. Ensembles of independently trained models were already used as a robustness device.

**Strengths.** Strong BraTS numbers; VAE regularization is a documented answer to small \(n\).  
**Limitations.** Ensembles multiply cost; CRF post-processing is inconsistent; winning a challenge year does not imply calibration or out-of-distribution robustness.

## 3. Self-configuring CNN pipelines

Isensee et al. (nnU-Net, *Nature Methods* 2021) argued that **pipeline configuration** (spacing, patch size, topology, augmentation, post-processing) often matters more than a novel block. nnU-Net self-configures a U-Net family and outperformed many specialized networks on 23 public tasks.

The 2024 follow-up, “nnU-Net Revisited,” reports that several recent 3D architectures **do not beat a modernized nnU-Net** when validation is held to the same standard. That paper is a warning against claiming architectural novelty from poorly controlled BraTS leaderboards.

MedNeXt (Roy et al., MICCAI 2023) modernizes 3D ConvNets with ConvNeXt-style large kernels and residual resampling. It is a strong **convolutional** alternative to transformers, not a claim that transformers are obsolete.

**Strengths.** Reproducible strong baseline; honest 2024 evidence that CNNs remain competitive in 3D.  
**Limitations.** Full nnU-Net is heavy to retrain; it is a system, not a single loss or uncertainty method. Dice-centric reporting still dominates.

## 4. Attention U-Net

Oktay et al. (MIDL 2018) add **attention gates** on skip connections so the decoder can suppress irrelevant encoder features. The original experiments are abdominal CT, not glioma MRI. The mechanism is architecture-agnostic and cheap.

**Strengths.** Small compute overhead; interpretable skip gating.  
**Limitations.** Not a 3D glioma paper; attention maps are not calibrated probabilities; gains over a well-tuned 3D U-Net on BraTS are not established in that paper.

## 5. Transformer-based segmentation

Chen et al. (TransUNet, 2021, arXiv) tokenize CNN feature maps and run a ViT encoder with a U-Net decoder (2D organs/cardiac). Wang et al. (TransBTS, MICCAI 2021) apply a 3D CNN encoder, a Transformer bottleneck, and a 3D decoder **on BraTS 2019/2020**. Hatamizadeh et al. (UNETR, WACV 2022) use a ViT encoder on 3D patches. Swin UNETR (BrainLes 2021 / LNCS 2022) uses shifted-window transformers specifically for **brain tumor MRI**.

**Strengths.** Explicit long-range context; Swin UNETR is a genuine glioma MRI paper with public code.  
**Limitations.** High VRAM; 2D TransUNet results do not transfer automatically to 3D MRI; Isensee et al. (2024) caution that transformer gains often vanish under matched preprocessing. Global attention does not by itself fix ET boundaries or calibration.

## 6. Uncertainty estimation

Gal and Ghahramani (ICML 2016) interpret dropout at test time as approximate Bayesian inference (MC dropout). Kendall and Gal (NeurIPS 2017) separate **aleatoric** (data) and **epistemic** (model) uncertainty. Lakshminarayanan et al. (NeurIPS 2017) show that **deep ensembles** give well-calibrated predictive distributions without a fully Bayesian training loop.

On **brain tumors**, Jungo et al. (*Frontiers in Neuroscience*, 2020) compared common uncertainty estimators. They found:

- dataset-level calibration can look acceptable;
- **subject-level** calibration is often poor;
- voxel-wise maps are weak for local error correction;
- aggregating uncertainty to the **patient** level is more useful for detecting failed cases.

Mehrtash et al. (*IEEE TMI*, 2020) studied calibration of segmentation networks more generally (not glioma-only) and argued that confidence must be evaluated with reliability diagrams and expected calibration error (ECE), not only Dice.

**Strengths.** MC dropout and TTA are cheap relative to ensembles; patient-level failure detection is supported.  
**Limitations.** Voxel-wise “uncertainty” is not automatically a trustworthy error map; dropout placement is under-specified; ensembles are expensive.

## 7. Boundary-aware learning

Regional losses (cross-entropy, Dice, generalized Dice) integrate over volumes. For thin or rare classes, a few distant false positives barely change Dice but dominate Hausdorff distance.

Kervadec et al. (MIDL 2019; *Medical Image Analysis* 2021) define a **boundary loss** as a regional integral of network softmax probabilities against a ground-truth **signed distance map**. Combined with generalized Dice, they reported Dice and Hausdorff gains on highly unbalanced lesions (ISLES, WMH), not BraTS specifically.

Karimi and Salcudean (*IEEE TMI*, 2020) propose losses that **directly approximate Hausdorff distance** via distance transforms, morphological erosion, or spherical kernels, with reported HD reductions of about 18–45% without harming Dice on prostate, liver, and pancreas.

**Strengths.** Complements Dice; targets the metric that fails on ET speckles.  
**Limitations.** Distance maps assume a reasonably good initial contour; not yet the BraTS default; can over-smooth if weighted too high.

## 8. Benchmarks that any NeuroVision-X study must cite

Menze et al. (*IEEE TMI*, 2015) define BraTS and document **human inter-rater Dice roughly 74–85%** depending on sub-region. Bakas et al. (*Scientific Data*, 2017) release expert-revised TCGA glioma labels used in later BraTS releases. Challenge data are **not** a clinical trial dataset; they are preprocessed, skull-stripped, and co-registered.

## 9. Cross-cutting limitations (supported by the papers above)

1. **Dice saturation vs residual surface error.** Overlap-centric training under-penalizes small, distant mistakes (Karimi and Salcudean; Kervadec et al.).  
2. **Architecture claims vs pipeline effects.** nnU-Net and nnU-Net Revisited show that unmatched preprocessing inflates “SOTA” stories.  
3. **Uncertainty is not reliability.** Jungo et al. show voxel-wise maps are poorly calibrated per patient.  
4. **ET is the brittle class.** Nested evaluation hides ET failures inside WT Dice.  
5. **No clinical validation** in these algorithmic papers. Challenge Dice is not diagnostic performance.

## 10. Paper catalogue (verified)

| # | Paper | Year | Venue | Identifier |
| --- | --- | --- | --- | --- |
| 1 | Ronneberger, Fischer, Brox. *U-Net: Convolutional Networks for Biomedical Image Segmentation* | 2015 | MICCAI, LNCS 9351, 234–241 | [DOI 10.1007/978-3-319-24574-4_28](https://doi.org/10.1007/978-3-319-24574-4_28) |
| 2 | Çiçek, Abdulkadir, Lienkamp, Brox, Ronneberger. *3D U-Net: Learning Dense Volumetric Segmentation from Sparse Annotation* | 2016 | MICCAI, LNCS 9901, 424–432 | [DOI 10.1007/978-3-319-46723-8_49](https://doi.org/10.1007/978-3-319-46723-8_49) |
| 3 | Milletari, Navab, Ahmadi. *V-Net: Fully Convolutional Neural Networks for Volumetric Medical Image Segmentation* | 2016 | 3DV, 565–571 | [DOI 10.1109/3DV.2016.79](https://doi.org/10.1109/3DV.2016.79) |
| 4 | Menze et al. *The Multimodal Brain Tumor Image Segmentation Benchmark (BRATS)* | 2015 | IEEE TMI 34(10), 1993–2024 | [DOI 10.1109/TMI.2014.2377694](https://doi.org/10.1109/TMI.2014.2377694) |
| 5 | Bakas et al. *Advancing The Cancer Genome Atlas glioma MRI collections with expert segmentation labels and radiomic features* | 2017 | Scientific Data 4:170117 | [DOI 10.1038/sdata.2017.117](https://doi.org/10.1038/sdata.2017.117) |
| 6 | Kamnitsas et al. *Efficient Multi-Scale 3D CNN with Fully Connected CRF for Accurate Brain Lesion Segmentation* | 2017 | Medical Image Analysis 36, 61–78 | [DOI 10.1016/j.media.2016.10.004](https://doi.org/10.1016/j.media.2016.10.004) |
| 7 | Myronenko. *3D MRI Brain Tumor Segmentation Using Autoencoder Regularization* | 2019 | BrainLes 2018, LNCS, 311–320 | [DOI 10.1007/978-3-030-11726-9_28](https://doi.org/10.1007/978-3-030-11726-9_28) |
| 8 | Oktay et al. *Attention U-Net: Learning Where to Look for the Pancreas* | 2018 | MIDL 2018 | [arXiv:1804.03999](https://arxiv.org/abs/1804.03999), [DOI 10.48550/arXiv.1804.03999](https://doi.org/10.48550/arXiv.1804.03999). OpenReview: https://openreview.net/forum?id=Skft7cijM |
| 9 | Gal, Ghahramani. *Dropout as a Bayesian Approximation: Representing Model Uncertainty in Deep Learning* | 2016 | ICML, PMLR 48, 1050–1059 | Official: https://proceedings.mlr.press/v48/gal16.html (dblp: no Crossref DOI). Preprint [DOI 10.48550/arXiv.1506.02142](https://doi.org/10.48550/arXiv.1506.02142) |
| 10 | Kendall, Gal. *What Uncertainties Do We Need in Bayesian Deep Learning for Computer Vision?* | 2017 | NeurIPS | Official: https://proceedings.neurips.cc/paper/2017/hash/2650d6089a6d640c5e85b2b88265dc2b-Abstract.html . Preprint [DOI 10.48550/arXiv.1703.04977](https://doi.org/10.48550/arXiv.1703.04977) |
| 11 | Lakshminarayanan, Pritzel, Blundell. *Simple and Scalable Predictive Uncertainty Estimation using Deep Ensembles* | 2017 | NeurIPS | Official: https://proceedings.neurips.cc/paper/2017/hash/9ef2ed4b7fd2c810847ffa5fa85bce38-Abstract.html . Preprint [DOI 10.48550/arXiv.1612.01474](https://doi.org/10.48550/arXiv.1612.01474) |
| 12 | Isensee, Jaeger, Kohl, Petersen, Maier-Hein. *nnU-Net: a self-configuring method for deep learning-based biomedical image segmentation* | 2021 | Nature Methods 18, 203–211 | [DOI 10.1038/s41592-020-01008-z](https://doi.org/10.1038/s41592-020-01008-z) |
| 13 | Chen et al. *TransUNet: Transformers Make Strong Encoders for Medical Image Segmentation* | 2021 | arXiv preprint | [arXiv:2102.04306](https://arxiv.org/abs/2102.04306), [DOI 10.48550/arXiv.2102.04306](https://doi.org/10.48550/arXiv.2102.04306) |
| 14 | Wang, Chen, Ding, Yu, Zha, Li. *TransBTS: Multimodal Brain Tumor Segmentation Using Transformer* | 2021 | MICCAI, LNCS 12901, 109–119 | [DOI 10.1007/978-3-030-87193-2_11](https://doi.org/10.1007/978-3-030-87193-2_11) |
| 15 | Hatamizadeh et al. *UNETR: Transformers for 3D Medical Image Segmentation* | 2022 | WACV | [DOI 10.1109/WACV51458.2022.00181](https://doi.org/10.1109/WACV51458.2022.00181); OA: https://openaccess.thecvf.com/content/WACV2022/html/Hatamizadeh_UNETR_Transformers_for_3D_Medical_Image_Segmentation_WACV_2022_paper.html |
| 16 | Hatamizadeh, Nath, Tang, Yang, Roth, Xu. *Swin UNETR: Swin Transformers for Semantic Segmentation of Brain Tumors in MRI Images* | 2022 | BrainLes 2021, LNCS 12962, 272–284 | [DOI 10.1007/978-3-031-08999-2_22](https://doi.org/10.1007/978-3-031-08999-2_22) |
| 17 | Kervadec et al. *Boundary loss for highly unbalanced segmentation* | 2021 | Medical Image Analysis 67:101851 | [DOI 10.1016/j.media.2020.101851](https://doi.org/10.1016/j.media.2020.101851) (MIDL 2019 precursor: https://proceedings.mlr.press/v102/kervadec19a.html) |
| 18 | Karimi, Salcudean. *Reducing the Hausdorff Distance in Medical Image Segmentation With Convolutional Neural Networks* | 2020 | IEEE TMI 39(2), 499–513 | [DOI 10.1109/TMI.2019.2930068](https://doi.org/10.1109/TMI.2019.2930068) |
| 19 | Jungo et al. *Analyzing the Quality and Challenges of Uncertainty Estimations for Brain Tumor Segmentation* | 2020 | Frontiers in Neuroscience 14:282 | [DOI 10.3389/fnins.2020.00282](https://doi.org/10.3389/fnins.2020.00282) |
| 20 | Mehrtash, Wells, Tempany, Abolmaesumi, Kapur. *Confidence Calibration and Predictive Uncertainty Estimation for Deep Medical Image Segmentation* | 2020 | IEEE TMI 39(12), 3868–3878 | [DOI 10.1109/TMI.2020.3006437](https://doi.org/10.1109/TMI.2020.3006437) |
| 21 | Roy, Koehler, Ulrich, Baumgartner, Petersen, Isensee, Jaeger, Maier-Hein. *MedNeXt: Transformer-Driven Scaling of ConvNets for Medical Image Segmentation* | 2023 | MICCAI, LNCS 14223, 405–415 | [DOI 10.1007/978-3-031-43901-8_39](https://doi.org/10.1007/978-3-031-43901-8_39) |
| 22 | Isensee, Wald, Ulrich, Baumgartner, Roy, Maier-Hein, Jäger. *nnU-Net Revisited: A Call for Rigorous Validation in 3D Medical Image Segmentation* | 2024 | MICCAI, LNCS, 488–498 | [DOI 10.1007/978-3-031-72114-4_47](https://doi.org/10.1007/978-3-031-72114-4_47) |

TransUNet and Attention U-Net are included because they are the canonical sources of those methods; they are **not** 3D glioma papers. That distinction is kept in the gap analysis.

See `references.bib` / `references.md` for BibTeX and IEEE-style entries. Phase 4–6 additionally cite Hu et al. (SE, CVPR 2018), Selvaraju et al. (Grad-CAM, ICCV 2017), Zeiler and Fergus (ECCV 2014), Adebayo et al. (NeurIPS 2018), and Brier (1950); those identifiers are in the same bibliography. They are **not** 3D glioma method papers except as used for fusion, attribution, and scoring rules.
