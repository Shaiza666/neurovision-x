# Explainability and visualization (Phase 6)

**This software is a research prototype, not a clinically validated diagnostic tool.**  
Heatmaps, overlays, and 3D previews **do not** prove that the model reasoned correctly, that a voxel “caused” a prediction, or that a radiologist should trust the output.

Figures are generated only from a **forward (and, for Grad-CAM, backward) pass on the tensor that was loaded**. They are not painted from challenge leaderboards or from hand-edited masks.

## What is implemented

| Product | Method | Citation / status |
| --- | --- | --- |
| 3D Grad-CAM | Gradient-weighted activations of a Conv3d layer, ReLU, trilinear upsample | Selvaraju et al., ICCV 2017 (2D); 3D is the same identity with 3D GAP and interpolation |
| Occlusion sensitivity | Drop cubes, record drop in class mass | Zeiler & Fergus, ECCV 2014 (style); `occlusion.py`, research dashboard (small volumes), `scripts/explain.py --also-occlusion` |
| Input gradient | \(\lvert \partial s / \partial x \rvert\) averaged over modalities | Standard saliency; not a new method |
| Uncertainty map | Predictive entropy of MC dropout or of a single softmax | Gal & Ghahramani 2016; Jungo et al. 2020. Dashboard: T=1 always; optional MC on small volumes. Flip TTA: `analyze_reliability.py` |
| Overlays | Predicted labels on MRI; GT vs pred when a seg exists; FP/FN error | Linked to \(\arg\max\) of the same logits as Grad-CAM |
| Orthogonal views | Axial / coronal / sagittal = spatial axes 0 / 1 / 2 of the RAS array | Labelled as tensor axes, not DICOM instance numbers |
| 3D preview | Subsampled point cloud of predicted WT/ET | Matplotlib 3D, not a clinical renderer |

Default Grad-CAM **score** is the mean logit of class \(c\) on voxels the model already labelled as \(c\). That ties the map to the **actual segmentation**, not to an unrelated class probe. If class \(c\) is empty, the fallback is the mean logit of \(c\) over the volume (documented in the JSON sidecar).

## Colour and labels

- MRI: grayscale.  
- Classes: background black, NCR blue, ED orange, ET red, with an on-figure legend.  
- Grad-CAM: sequential magma, colourbar “Grad-CAM (normalised, unitless)”.  
- Entropy: sequential viridis, colourbar “predictive entropy (nats)”.  
- Error: TP green, FP red, FN blue; legend required.  
- Every figure title includes that this is research QC, not a clinical explanation.

## Limitations of Grad-CAM (must be stated with every use)

1. **Not a causal explanation.** Grad-CAM is a coarse localization of gradient-weighted feature maps. It does not identify which voxels were necessary or sufficient for the decision (see also Adebayo et al., NeurIPS 2018, on saliency sanity checks).  
2. **Low spatial precision.** The map lives on the chosen Conv3d grid and is **interpolated** to voxel size. Bright blobs are larger than the true receptive-field story.  
3. **Target-score dependence.** Changing the scalar \(s\) (predicted-region mean vs global class mean vs logit at one voxel) changes the map. We log the score mode; we do not search for a flattering \(s\).  
4. **Layer dependence.** A 1×1 head is a poor CAM target; we default to the last spatial \(k>1\) Conv3d. A different layer is a different picture, not “the” explanation.  
5. **Can highlight the tumor because the decoder already segmented it.** For segmentation, CAM on the last decoder often traces the prediction itself. That is **not** independent evidence that the prediction is correct.  
6. **Unstable under input shifts.** Small intensity changes, dropout, and AMP can move the map. Do not treat a single PNG as a reproducibility unit without the JSON sidecar (layer name, class index, score).  
7. **Not validated against radiologist gaze or pathology.** No eye-tracking or histology study is claimed.

## Limitations of uncertainty visualizations

1. **Entropy is not error.** High entropy can mark tissue boundaries, rare appearances, or dropout noise. Jungo et al. (2020) found **patient-level** aggregation more useful for failed glioma cases than voxel maps for local correction.  
2. **MC dropout is an approximation** (Gal & Ghahramani, 2016), not a calibrated posterior. ECE/Brier belong in Phase 5 JSON, not in a colour overlay.  
3. **A confident voxel can still be wrong** (overconfidence). A red entropy blob is not a biopsy.  
4. **Overlays can look “explanatory” while only restating the softmax.** Showing entropy next to the prediction does not add clinical reasoning.

## Occlusion and gradients

Occlusion is model-agnostic but **out-of-distribution** (zero cubes are not MRI). Input gradients are local linearizations and can be noisy. Neither is a clinical rationale.

## What this module will not do

- Will not emit a BraTS patient figure unless that patient’s NIfTI and a **non-smoke** checkpoint were actually loaded.  
- Will not copy heatmaps from papers.  
- Will not use synthetic MRI as a published case study (`--synthetic` is labelled as such).  
- Will not claim that overlap between CAM and tumor proves the model is correct.

## How to run (after data and a real checkpoint)

```powershell
.\.venv\Scripts\python scripts\explain.py --run-name A0_baseline --patient-id <id>
```

Without BraTS folders the script exits. `--synthetic` writes a watermarked layout demo from random tensors (not MRI). An example of that layout (random cubes, **not** a patient) is in `docs/figures/explain_synthetic_not_mri/`.
