"""Quality-control figures from real (or unit-test fixture) volumes.

Figures produced here are research QC, not clinical images. They are only
written for volumes that were actually loaded.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from neurovisionx.evaluation.regions import nested_masks


def _require_matplotlib():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def _mid(length: int) -> int:
    return max(length // 2, 0)


def save_patient_qc(
    image: np.ndarray,
    seg: np.ndarray,
    destination: Path,
    *,
    title: str,
    channel_names: tuple[str, ...] = ("t1", "t1ce", "t2", "flair"),
) -> Path:
    """Save a 2x2 of mid-slice MRI with WT/ET overlay on the last panel."""
    plt = _require_matplotlib()
    destination.parent.mkdir(parents=True, exist_ok=True)
    if image.ndim != 4:
        raise ValueError("image must be (C, D, H, W)")
    depth, height, width = image.shape[1:]
    z, y, x = _mid(depth), _mid(height), _mid(width)
    masks = nested_masks(seg)
    n_show = min(3, image.shape[0])
    fig, axes = plt.subplots(2, 2, figsize=(8, 8))
    panels = [
        (axes[0, 0], image[min(1, n_show - 1), z], "axial"),
        (axes[0, 1], image[min(1, n_show - 1), :, y, :], "coronal"),
        (axes[1, 0], image[min(3, n_show - 1) if image.shape[0] > 3 else 0, z], "axial-flair"),
        (axes[1, 1], image[min(3, n_show - 1) if image.shape[0] > 3 else 0, z], "overlay"),
    ]
    for ax, plane, name in panels:
        ax.imshow(np.rot90(plane), cmap="gray")
        ax.set_title(name)
        ax.axis("off")
    overlay = axes[1, 1]
    wt = np.rot90(masks["WT"][z])
    et = np.rot90(masks["ET"][z])
    overlay.imshow(np.ma.masked_where(~wt, wt.astype(float)), cmap="Wistia", alpha=0.35)
    overlay.imshow(np.ma.masked_where(~et, et.astype(float)), cmap="autumn", alpha=0.5)
    fig.suptitle(title + "\nResearch QC — not a clinical image")
    fig.tight_layout()
    fig.savefig(destination, dpi=120)
    plt.close(fig)
    return destination


def save_class_histogram(counts: dict[str, int], destination: Path) -> Path:
    plt = _require_matplotlib()
    destination.parent.mkdir(parents=True, exist_ok=True)
    labels = list(counts.keys())
    values = [counts[name] for name in labels]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(labels, values, color="#3b6ea5")
    ax.set_ylabel("voxels")
    ax.set_title("Class / region voxel counts (research QC)")
    fig.tight_layout()
    fig.savefig(destination, dpi=120)
    plt.close(fig)
    return destination
