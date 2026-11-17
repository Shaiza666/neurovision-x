"""Overlap, surface, and calibration metrics.

Dice and HD95 follow BraTS nested-region reporting. ECE follows Mehrtash
et al., IEEE TMI 2020. Empty predictions yield Dice 0 and HD95 undefined
(returned as NaN), never a fabricated score.
"""

from __future__ import annotations

import math

import numpy as np


def dice_coefficient(pred: np.ndarray, target: np.ndarray) -> float:
    pred_b = pred.astype(bool)
    target_b = target.astype(bool)
    intersection = np.logical_and(pred_b, target_b).sum()
    denom = pred_b.sum() + target_b.sum()
    if denom == 0:
        return 1.0 if intersection == 0 else 0.0
    return float(2 * intersection / denom)


def _surface_distances(pred: np.ndarray, target: np.ndarray) -> np.ndarray:
    try:
        from scipy.ndimage import binary_erosion, distance_transform_edt
    except ImportError as exc:
        raise ImportError("scipy is required for HD95") from exc

    def _surface(mask: np.ndarray) -> np.ndarray:
        eroded = binary_erosion(mask, border_value=0)
        return np.logical_and(mask, np.logical_not(eroded))

    pred_b = pred.astype(bool)
    target_b = target.astype(bool)
    if not pred_b.any() or not target_b.any():
        return np.array([math.nan], dtype=np.float64)
    pred_surf = _surface(pred_b)
    target_surf = _surface(target_b)
    dt_target = distance_transform_edt(~target_b)
    dt_pred = distance_transform_edt(~pred_b)
    d_pred = dt_target[pred_surf]
    d_target = dt_pred[target_surf]
    return np.concatenate([d_pred, d_target])


def hd95(pred: np.ndarray, target: np.ndarray) -> float:
    distances = _surface_distances(pred, target)
    if distances.size == 0 or np.isnan(distances).all():
        return float("nan")
    finite = distances[np.isfinite(distances)]
    if finite.size == 0:
        return float("nan")
    return float(np.percentile(finite, 95))


def expected_calibration_error(
    confidence: np.ndarray,
    correct: np.ndarray,
    n_bins: int = 15,
) -> float:
    """ECE on flattened confidence/correct arrays (Mehrtash et al., 2020)."""
    confidence = np.asarray(confidence, dtype=np.float64).ravel()
    correct = np.asarray(correct, dtype=np.float64).ravel()
    if confidence.size == 0:
        return float("nan")
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n_edges = len(bins) - 1
    for i, (left, right) in enumerate(zip(bins[:-1], bins[1:])):
        if i == n_edges - 1:
            inside = (confidence >= left) & (confidence <= right)
        else:
            inside = (confidence >= left) & (confidence < right)
        if not np.any(inside):
            continue
        acc = correct[inside].mean()
        conf = confidence[inside].mean()
        ece += (inside.mean()) * abs(acc - conf)
    return float(ece)


def negative_log_likelihood(
    probs: np.ndarray,
    target: np.ndarray,
    eps: float = 1e-8,
) -> float:
    """Mean NLL of integer labels under a softmax distribution.

    ``probs`` is (C, ...) matching ``target`` spatial shape. Returns NaN if
    empty. This is a calibration diagnostic, not a clinical score.
    """
    probs = np.asarray(probs, dtype=np.float64)
    target = np.asarray(target, dtype=np.int64)
    if probs.ndim != target.ndim + 1:
        raise ValueError("probs must have a leading class axis over target's shape")
    if probs.size == 0:
        return float("nan")
    clipped = np.clip(probs, eps, 1.0)
    gathered = np.take_along_axis(clipped, target[None, ...], axis=0)[0]
    return float(-np.log(gathered).mean())
