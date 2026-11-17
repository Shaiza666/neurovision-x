# Research hypotheses

**Project:** NeuroVision-X, Phase 1  
**Status:** hypotheses only. They are not results. They will be accepted or rejected only after a real training run on official BraTS data with patient-level splits.

This software is not a clinically validated diagnostic tool.

Notation: \(G\) ground-truth mask, \(P\) prediction, nested BraTS regions WT / TC / ET. Soft Dice for class \(c\):

\[
\mathrm{Dice}_c = \frac{2\sum_i p_{i,c} g_{i,c} + \varepsilon}{\sum_i p_{i,c} + \sum_i g_{i,c} + \varepsilon}.
\]

HD95 is the 95th percentile of bidirectional surface distances. Expected calibration error (ECE) bins predicted confidence against empirical accuracy (Mehrtash et al., 2020).

Boundary loss follows Kervadec et al. (2021): if \(\phi_G\) is the signed distance function of \(G\) and \(s_\theta\) is the softmax foreground map,

\[
\mathcal{L}_{\mathrm{bd}}(\theta) = \int_\Omega s_\theta(x)\,\phi_G(x)\,dx
\]

(up to implementation constants). Combined objective:

\[
\mathcal{L} = \lambda_{\mathrm{dice}}\mathcal{L}_{\mathrm{dice}} + \lambda_{\mathrm{ce}}\mathcal{L}_{\mathrm{ce}} + \lambda_{\mathrm{bd}}\mathcal{L}_{\mathrm{bd}}.
\]

MC dropout (Gal and Ghahramani, 2016) approximates the posterior predictive by \(T\) stochastic forward passes. Voxel entropy of the mean softmax is the uncertainty map; the **patient score** is the mean entropy over WT-candidate voxels.

## H1 — Boundary-aware loss reduces ET surface error without sacrificing overlap

**Statement.** On a held-out BraTS patient split, a 3D U-Net trained with Dice+CE+boundary loss achieves **lower mean ET HD95** than the same architecture trained with Dice+CE only, with **ET Dice not worse** (difference within a pre-registered non-inferiority margin, e.g. 1 Dice point).

**Rationale.** Dice is region-based; ET is small; HD95 is driven by outliers (Karimi and Salcudean, 2020; Kervadec et al., 2021).

**Falsifiers.** No HD95 improvement; Dice drop larger than the margin; instability of \(\lambda_{\mathrm{bd}}\).

**Established method:** boundary / HD losses. **What is being tested:** whether that transfer holds on **glioma ET** under a matched 3D U-Net.

## H2 — Patient-aggregated uncertainty predicts segmentation failure better than voxel-wise maps

**Statement.** Mean MC-dropout (or TTA) entropy per patient correlates with that patient’s WT/TC/ET Dice (Spearman \(\rho < 0\), pre-registered \(\alpha=0.05\)), and a simple threshold on that score detects the worst quartile of cases with higher AUPRC than using the mean voxel-wise entropy as a local error locator (Jungo-style error-localization analysis).

**Rationale.** Jungo et al. (2020) found voxel-wise maps poorly suited to local correction but useful when **aggregated**. Mehrtash et al. (2020) require calibration metrics, not Dice alone.

**Falsifiers.** No correlation; ECE worse than the deterministic baseline; dropout entropy uncorrelated with error.

**Established method:** MC dropout / TTA. **What is being tested:** **patient-level** failure detection on BraTS, not a new Bayesian layer.

## H3 — Under matched training, a 3D CNN baseline is competitive with a 3D transformer of similar compute

**Statement.** A MONAI 3D U-Net / DynUNet trained with the same patch size, augmentations, epochs, and split achieves WT/TC/ET Dice within 1 point of UNETR or Swin UNETR at similar parameter/FLOP budget on the **same** validation patients.

**Rationale.** Isensee et al. (2024) and MedNeXt (Roy et al., 2023) argue that CNN pipelines remain strong in 3D; transformer papers (Wang et al., 2021; Hatamizadeh et al., 2022) often change many variables at once.

**Falsifiers.** Transformer wins by a clear margin under the matched protocol; CNN cannot fit in memory while the transformer can (then H3 is **not tested**, it is a compute constraint).

**If GPU memory forbids a transformer, H3 is deferred.** That will be reported as a resource limit, not as a negative result against transformers.

## Testing protocol (all three)

- Official BraTS MRI only after license/registration. No synthetic volumes in reported metrics.
- Patient-level train/val/test. No slice-level shuffling across patients.
- Fixed seed, logged config, identical preprocessing for compared runs.
- Metrics: Dice and HD95 on WT, TC, ET; ECE/NLL if H2 is run.
- Unsuccessful runs are reported. Training is “complete” only after an actual job finishes.

## Priority if compute is limited

1. H1 on 3D U-Net (loss ablation).  
2. H2 on the better of the H1 models.  
3. H3 only if a second architecture fits.
