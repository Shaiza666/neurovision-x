"""BraTS nested regions from voxel class maps.

Official nested evaluation (Menze et al., 2015; Bakas / BraTS reporting):

- WT (whole tumor) = NCR ∪ ED ∪ ET
- TC (tumor core) = NCR ∪ ET
- ET (enhancing tumor) = ET

Legacy BraTS labels used 4 for ET (3 unused). Newer releases use 3 for ET.
Both are remapped here. These helpers do not invent labels for missing volumes.
"""

from __future__ import annotations

import numpy as np

from neurovisionx.evaluation.metrics import dice_coefficient, hd95

ET_LEGACY = 4
NCR, ED, ET = 1, 2, 3
REGION_NAMES = ("WT", "TC", "ET")


def remap_brats_labels(label: np.ndarray) -> np.ndarray:
    """Map legacy ET label 4 to 3. Other integers are unchanged."""
    remapped = np.array(label, copy=True)
    remapped[remapped == ET_LEGACY] = ET
    return remapped


def nested_masks(label: np.ndarray) -> dict[str, np.ndarray]:
    """Return boolean WT/TC/ET masks from a class map."""
    lab = remap_brats_labels(label)
    return {
        "WT": np.isin(lab, (NCR, ED, ET)),
        "TC": np.isin(lab, (NCR, ET)),
        "ET": lab == ET,
    }


def region_scores(
    pred: np.ndarray,
    target: np.ndarray,
    *,
    compute_hd95: bool = False,
) -> dict[str, float]:
    """Dice (and optional HD95) for nested regions.

    Empty prediction vs empty target is Dice 1. Empty vs non-empty is Dice 0.
    HD95 is NaN when a surface cannot be formed. No placeholder scores.
    """
    pred_n = nested_masks(pred)
    true_n = nested_masks(target)
    scores: dict[str, float] = {}
    for name in REGION_NAMES:
        key = name.lower()
        scores[f"dice_{key}"] = dice_coefficient(pred_n[name], true_n[name])
        if compute_hd95:
            scores[f"hd95_{key}"] = hd95(pred_n[name], true_n[name])
    return scores
