"""Calibration tables and NLL (Mehrtash et al., IEEE TMI 2020).

These quantities are research diagnostics. They are not clinical confidence
scores. Values from synthetic tensors must not be reported as BraTS results.
"""

from __future__ import annotations

import numpy as np

from neurovisionx.evaluation.metrics import expected_calibration_error, negative_log_likelihood


def reliability_table(
    confidence: np.ndarray,
    correct: np.ndarray,
    n_bins: int = 15,
) -> list[dict[str, float]]:
    """Per-bin accuracy vs confidence for a reliability diagram."""
    confidence = np.asarray(confidence, dtype=np.float64).ravel()
    correct = np.asarray(correct, dtype=np.float64).ravel()
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    rows: list[dict[str, float]] = []
    n_edges = len(bins) - 1
    for i, (left, right) in enumerate(zip(bins[:-1], bins[1:])):
        if i == n_edges - 1:
            inside = (confidence >= left) & (confidence <= right)
        else:
            inside = (confidence >= left) & (confidence < right)
        count = int(inside.sum())
        if count == 0:
            rows.append(
                {
                    "left": float(left),
                    "right": float(right),
                    "count": 0.0,
                    "accuracy": float("nan"),
                    "confidence": float("nan"),
                }
            )
            continue
        rows.append(
            {
                "left": float(left),
                "right": float(right),
                "count": float(count),
                "accuracy": float(correct[inside].mean()),
                "confidence": float(confidence[inside].mean()),
            }
        )
    return rows


def calibration_summary(
    confidence: np.ndarray,
    correct: np.ndarray,
    probs: np.ndarray | None = None,
    target: np.ndarray | None = None,
    n_bins: int = 15,
) -> dict[str, float | list]:
    payload: dict[str, float | list] = {
        "ece": expected_calibration_error(confidence, correct, n_bins=n_bins),
        "n_voxels": int(np.asarray(confidence).size),
        "n_bins": n_bins,
        "reliability": reliability_table(confidence, correct, n_bins=n_bins),
    }
    if probs is not None and target is not None:
        payload["nll"] = negative_log_likelihood(probs, target)
    return payload
