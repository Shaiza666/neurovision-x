"""Research QC plots for uncertainty and calibration.

Overlays are not explanations for clinicians and are not diagnostic images.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np


def _require_matplotlib():
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise ImportError("matplotlib is required to write uncertainty figures") from exc
    return plt


def mid_slice(volume: np.ndarray, axis: int = 0) -> np.ndarray:
    """Return the middle 2D slice of a 3D array."""
    array = np.asarray(volume)
    if array.ndim != 3:
        raise ValueError("expected a 3D volume")
    index = array.shape[axis] // 2
    return np.take(array, index, axis=axis)


def save_uncertainty_overlay(
    image: np.ndarray,
    prediction: np.ndarray,
    entropy: np.ndarray,
    error: np.ndarray,
    path: Path | str,
    title: str = "Uncertainty overlay (research QC, not clinical)",
) -> Path:
    """Write a 2x2 figure: image, prediction, entropy, error mask."""
    plt = _require_matplotlib()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(8, 8))
    panels = (
        (axes[0, 0], image, "image", "gray"),
        (axes[0, 1], prediction, "prediction", "nipy_spectral"),
        (axes[1, 0], entropy, "predictive entropy", "magma"),
        (axes[1, 1], error.astype(float), "error", "Reds"),
    )
    for ax, data, label, cmap in panels:
        ax.imshow(np.asarray(data), cmap=cmap)
        ax.set_title(label)
        ax.axis("off")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


def save_reliability_diagram(
    bins: list[dict],
    path: Path | str,
    title: str = "Reliability diagram (research, not clinical)",
) -> Path:
    plt = _require_matplotlib()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    conf = [row["confidence"] for row in bins if row["count"] > 0 and row["confidence"] == row["confidence"]]
    acc = [row["accuracy"] for row in bins if row["count"] > 0 and row["accuracy"] == row["accuracy"]]
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="perfect")
    if conf:
        ax.plot(conf, acc, marker="o", label="model")
    ax.set_xlabel("confidence")
    ax.set_ylabel("accuracy")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path
