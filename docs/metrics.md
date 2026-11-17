# Segmentation metrics

This software is a research prototype, not a clinically validated diagnostic tool.

Nested BraTS regions follow Menze et al. (2015): WT = NCR ∪ ED ∪ ET, TC = NCR ∪ ET, ET = enhancing tumor. Legacy label `4` is remapped to `3` before scoring (`src/neurovisionx/evaluation/regions.py`).

## Dice

For binary masks \(P\) and \(G\):

\[
\mathrm{Dice} = \frac{2|P \cap G|}{|P| + |G|}
\]

Empty \(P\) and empty \(G\) score 1. Empty \(P\) vs non-empty \(G\) scores 0. Implementation: `dice_coefficient` in `src/neurovisionx/evaluation/metrics.py`. Training uses the soft (probabilistic) form of Milletari et al. (2016) as a loss, excluding background by default.

## HD95

Hausdorff distance at the 95th percentile is a **surface** error. It is the metric BraTS uses alongside Dice; it is not a NeuroVision-X invention.

**Implementation** (`hd95` in `src/neurovisionx/evaluation/metrics.py`):

1. Convert each region mask to a binary volume.
2. If either mask is empty, return **NaN** (undefined surface). Never substitute 0 or a large sentinel as a “score”.
3. Extract the object surface as voxels that remain after binary erosion (`scipy.ndimage.binary_erosion`).
4. Compute Euclidean distance transforms of the complements (`distance_transform_edt`).
5. Collect distances from predicted surface voxels to the ground-truth object, and from ground-truth surface voxels to the predicted object (bidirectional).
6. Report the **95th percentile** of the concatenated distances.

Units are **voxels**. After BraTS-style 1 mm isotropic resampling, one voxel ≈ 1 mm, which is how challenge numbers are usually quoted. If a volume was not resampled to 1 mm, do not label HD95 as millimetres.

HD95 is optional during training (`compute_hd95_on_val`, default false) because distance transforms on every patch slow the loop. Full-volume HD95 is computed in `scripts/evaluate.py` on real data after a checkpoint exists.

Synthetic smoke tests may call the same function on random cubes to check that the code runs. Those numbers are **not** research results.

## Calibration (ECE, NLL)

Expected calibration error (Mehrtash et al., 2020) bins voxel confidence \(\max_c p_c\) against correctness. Negative log-likelihood is the mean \(-\log p_{y}\) of the same softmax. Both are research diagnostics, not clinical confidence. Implementation: `expected_calibration_error` and `negative_log_likelihood` in `src/neurovisionx/evaluation/metrics.py`. Reliability tables live in `src/neurovisionx/uncertainty/calibration.py`.

## Uncertainty–error and small ET

Patient-level predictive entropy (MC dropout, Gal and Ghahramani, 2016; patient aggregation, Jungo et al., 2020) is correlated with Dice using Spearman \(\rho\). Detection of the worst ET Dice quartile uses AUPRC. Small/difficult ET is the **pre-registered** lowest quartile of ET voxel count (`src/neurovisionx/evaluation/hard_cases.py`). Do not re-define the stratum after seeing scores.

These analyses run in `scripts/analyze_reliability.py` only after a real (non-smoke) checkpoint and official volumes exist.

## Precision and recall

For a binary region mask, precision = TP / (TP+FP) and recall = TP / (TP+FN). Empty vs empty scores 1; empty prediction vs non-empty target scores 0 (same convention as Dice). Implementation: `precision_recall` in `src/neurovisionx/evaluation/metrics.py`. Nested WT/TC/ET precision/recall are optional keys from `region_scores(..., compute_pr=True)`.

## Brier score

The multiclass Brier score is the mean squared error between the softmax vector and the one-hot label (Brier, 1950). It is a proper scoring rule used here as a calibration diagnostic, not a clinical score. Implementation: `brier_score_multiclass`.

## Mean, SD, and 95% CI

Scientific tables use the **sample** standard deviation (ddof=1) and a Student-t interval when \(n \ge 2\) (`src/neurovisionx/evaluation/summarize.py`). A single patient or seed has no CI. Missing JSON is reported as **not measured**, never as 0.

## GPU memory

Peak allocated megabytes come from `torch.cuda.max_memory_allocated` during one forward (`measure_gpu_memory`). If CUDA is unavailable the field is recorded as not measured. This is not `nvidia-smi` reserved memory and not a clinical hardware claim.
