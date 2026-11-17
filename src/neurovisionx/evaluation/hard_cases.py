"""Small-tumor strata and uncertainty–error association.

Strata are pre-registered (lowest quartile of ET voxel count). They are not
chosen after seeing which subset looks better. Synthetic tensors in tests are
not BraTS cases.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from neurovisionx.evaluation.regions import ET, remap_brats_labels


def et_voxel_count(label: np.ndarray) -> int:
    return int((remap_brats_labels(label) == ET).sum())


def size_stratum(
    counts: np.ndarray,
    values: np.ndarray,
    q: float = 0.25,
) -> dict[str, Any]:
    """Split ``values`` into the lowest ``q`` of ``counts`` vs the rest."""
    counts = np.asarray(counts, dtype=np.float64)
    values = np.asarray(values, dtype=np.float64)
    if counts.size == 0:
        return {"threshold": float("nan"), "n_small": 0, "n_rest": 0, "small_mean": float("nan"), "rest_mean": float("nan")}
    threshold = float(np.quantile(counts, q))
    small = counts <= threshold
    rest = ~small

    def _mean(mask: np.ndarray) -> float:
        subset = values[mask]
        finite = subset[np.isfinite(subset)]
        if finite.size == 0:
            return float("nan")
        return float(finite.mean())

    return {
        "q": float(q),
        "threshold": threshold,
        "n_small": int(small.sum()),
        "n_rest": int(rest.sum()),
        "small_mean": _mean(small),
        "rest_mean": _mean(rest),
    }


def spearman_uncertainty_dice(
    uncertainty: np.ndarray,
    dice: np.ndarray,
) -> dict[str, float]:
    """H2 association: higher uncertainty should track lower Dice if H2 holds.

    Returns rho and two-sided p. Does not claim H2 until a real BraTS table
    is written.
    """
    x = np.asarray(uncertainty, dtype=np.float64)
    y = np.asarray(dice, dtype=np.float64)
    finite = np.isfinite(x) & np.isfinite(y)
    x, y = x[finite], y[finite]
    if x.size < 3:
        return {"rho": float("nan"), "pvalue": float("nan"), "n": float(x.size)}
    try:
        from scipy.stats import spearmanr
    except ImportError as exc:
        raise ImportError("scipy is required for Spearman correlation") from exc
    result = spearmanr(x, y)
    rho = getattr(result, "statistic", None)
    if rho is None:
        rho = result.correlation
    return {"rho": float(rho), "pvalue": float(result.pvalue), "n": float(x.size)}


def binary_auprc(y_score: np.ndarray, y_true: np.ndarray) -> float:
    """Area under the precision–recall curve. ``y_true`` is 1 for the positive class.

    Used to score detection of the worst Dice quartile from a failure score
    (patient uncertainty). Returns NaN if there are no positives.
    """
    scores = np.asarray(y_score, dtype=np.float64)
    labels = np.asarray(y_true, dtype=np.int32)
    finite = np.isfinite(scores)
    scores, labels = scores[finite], labels[finite]
    n_pos = int(labels.sum())
    if scores.size == 0 or n_pos == 0:
        return float("nan")
    order = np.argsort(-scores, kind="mergesort")
    labels = labels[order]
    tp = np.cumsum(labels)
    fp = np.cumsum(1 - labels)
    precision = tp / np.maximum(tp + fp, 1)
    recall = tp / float(n_pos)
    recall = np.concatenate([[0.0], recall])
    precision = np.concatenate([[float(n_pos > 0)], precision])
    if hasattr(np, "trapezoid"):
        return float(np.trapezoid(precision, recall))
    return float(np.trapz(precision, recall))


def worst_quartile_labels(dice: np.ndarray) -> np.ndarray:
    """1 for patients at or below the 25th percentile of Dice."""
    values = np.asarray(dice, dtype=np.float64)
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return np.zeros_like(values, dtype=np.int32)
    threshold = float(np.quantile(finite, 0.25))
    return (values <= threshold).astype(np.int32)


def uncertainty_error_report(
    patient_uncertainty: np.ndarray,
    dice: np.ndarray,
    et_counts: np.ndarray,
) -> dict[str, Any]:
    failure = worst_quartile_labels(dice)
    return {
        "spearman": spearman_uncertainty_dice(patient_uncertainty, dice),
        "auprc_worst_et_dice_quartile": binary_auprc(patient_uncertainty, failure),
        "et_size_stratum_dice": size_stratum(et_counts, dice, q=0.25),
        "n_patients": int(np.asarray(dice).size),
        "note": (
            "H2 is supported only if Spearman rho is negative and the AUPRC "
            "is reported from official BraTS patients, not synthetic tensors."
        ),
        "clinical_disclaimer": "Research prototype. Not a clinically validated diagnostic tool.",
    }


def voxel_error_mask(pred: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Boolean mismatch after BraTS label remap. Used for overlay QC."""
    return remap_brats_labels(pred) != remap_brats_labels(target)
