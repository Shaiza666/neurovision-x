"""Lightweight 3D preview of predicted regions.

This is a subsampled Matplotlib scatter, not a clinical renderer, not a
radiology PACS 3D view, and not a proof of segmentation quality.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from neurovisionx.evaluation.regions import nested_masks
from neurovisionx.explainability.overlays import CLASS_RGB, DISCLAIMER


def _require_matplotlib():
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise ImportError("matplotlib is required for 3D previews") from exc
    return plt


def _points(mask: np.ndarray, max_points: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    coords = np.argwhere(np.asarray(mask, dtype=bool))
    if coords.size == 0:
        return np.array([]), np.array([]), np.array([])
    if coords.shape[0] > max_points:
        rng = np.random.default_rng(2026)
        choice = rng.choice(coords.shape[0], size=max_points, replace=False)
        coords = coords[choice]
    return coords[:, 2], coords[:, 1], coords[:, 0]


def save_prediction_pointcloud(
    prediction: np.ndarray,
    path: Path | str,
    *,
    max_points: int = 4000,
    title: str = DISCLAIMER,
) -> Path:
    """Scatter of WT (blue) and ET (red) predicted voxels, downsampled."""
    plt = _require_matplotlib()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    masks = nested_masks(prediction)
    fig = plt.figure(figsize=(6.2, 5.6))
    ax = fig.add_subplot(111, projection="3d")
    x_wt, y_wt, z_wt = _points(masks["WT"], max_points)
    x_et, y_et, z_et = _points(masks["ET"], max(max_points // 3, 200))
    if x_wt.size:
        ax.scatter(x_wt, y_wt, z_wt, s=2, color=CLASS_RGB[1], label="predicted WT", alpha=0.25)
    if x_et.size:
        ax.scatter(x_et, y_et, z_et, s=4, color=CLASS_RGB[3], label="predicted ET", alpha=0.6)
    if x_wt.size == 0 and x_et.size == 0:
        ax.text2D(0.3, 0.5, "No predicted tumor voxels", transform=ax.transAxes)
    ax.set_xlabel("W (axis 2)")
    ax.set_ylabel("H (axis 1)")
    ax.set_zlabel("D (axis 0, axial)")
    ax.legend(loc="upper left", fontsize=8)
    ax.set_title("3D preview of predicted masks (subsampled)")
    fig.suptitle(title, fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path
