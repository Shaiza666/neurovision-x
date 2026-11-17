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
