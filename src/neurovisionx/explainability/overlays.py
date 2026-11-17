"""MRI overlays, GT vs prediction, and FP/FN error maps.

Colours are labelled on the figure. Overlays are research QC, not diagnostic
images and not proof that the segmentation is correct.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from neurovisionx.evaluation.regions import nested_masks, remap_brats_labels
from neurovisionx.explainability.views import ORIENTATION_ORDER, extract_slice, slice_index

DISCLAIMER = (
    "Research visualization. Not a clinical explanation. "
    "Overlays do not prove model correctness."
)

# NCR / ED / ET (BraTS nested classes 1, 2, 3)
CLASS_RGB = {
    0: (0.0, 0.0, 0.0),
    1: (0.12, 0.47, 0.71),  # NCR
    2: (1.00, 0.73, 0.47),  # ED
    3: (0.84, 0.15, 0.16),  # ET
}
CLASS_NAMES = {0: "background", 1: "NCR", 2: "ED", 3: "ET"}

# Error RGB: TP green, FP red, FN blue
TP_RGB = (0.13, 0.55, 0.13)
FP_RGB = (0.80, 0.10, 0.10)
FN_RGB = (0.12, 0.35, 0.75)


def _require_matplotlib():
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.lines import Line2D
        from matplotlib.patches import Patch
    except ImportError as exc:
        raise ImportError("matplotlib is required for explainability figures") from exc
    return plt, Patch, Line2D


def mri_plane(image: np.ndarray, orientation: str, index: int | None = None) -> np.ndarray:
    """Prefer FLAIR (channel 3) when the stack has 4 modalities."""
    array = np.asarray(image)
    if array.ndim == 4:
        channel = array[3] if array.shape[0] >= 4 else array[0]
    else:
        channel = array
    plane = extract_slice(channel, orientation, index)  # type: ignore[arg-type]
    finite = plane[np.isfinite(plane)]
    if finite.size == 0:
        return plane
    lo, hi = np.percentile(finite, (1.0, 99.0))
    if hi <= lo:
        hi = lo + 1.0
    return np.clip((plane - lo) / (hi - lo), 0.0, 1.0)


def labels_to_rgb(labels: np.ndarray) -> np.ndarray:
    lab = remap_brats_labels(np.asarray(labels, dtype=np.int64))
    rgb = np.zeros(lab.shape + (3,), dtype=np.float64)
    for value, colour in CLASS_RGB.items():
        rgb[lab == value] = colour
    return rgb


def error_rgb(pred: np.ndarray, target: np.ndarray, region: str = "WT") -> np.ndarray:
    pred_m = nested_masks(pred)[region]
    true_m = nested_masks(target)[region]
    rgb = np.zeros(pred_m.shape + (3,), dtype=np.float64)
    rgb[np.logical_and(pred_m, true_m)] = TP_RGB
    rgb[np.logical_and(pred_m, np.logical_not(true_m))] = FP_RGB
    rgb[np.logical_and(np.logical_not(pred_m), true_m)] = FN_RGB
    return rgb


def _add_class_legend(ax, handles_cls) -> None:
    ax.legend(handles=handles_cls, loc="lower right", fontsize=7, frameon=True)


def save_orthogonal_overlay(
    image: np.ndarray,
    prediction: np.ndarray,
    path: Path | str,
    *,
    target: np.ndarray | None = None,
    indices: dict[str, int] | None = None,
    title: str = DISCLAIMER,
) -> Path:
    """3 orientations × MRI | prediction | GT or placeholder | WT error."""
    plt, Patch, Line2D = _require_matplotlib()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pred = remap_brats_labels(prediction)
    true = remap_brats_labels(target) if target is not None else None
    fig, axes = plt.subplots(3, 4, figsize=(12.5, 9.2))
    class_handles = [
        Patch(facecolor=CLASS_RGB[i], edgecolor="none", label=CLASS_NAMES[i]) for i in (0, 1, 2, 3)
    ]
    error_handles = [
        Patch(facecolor=TP_RGB, label="TP (WT)"),
        Patch(facecolor=FP_RGB, label="FP (WT)"),
        Patch(facecolor=FN_RGB, label="FN (WT)"),
    ]
    contour_handles = [
        Line2D([0], [0], color=CLASS_RGB[3], lw=1.5, label="ET contour (pred)"),
        Line2D([0], [0], color=CLASS_RGB[1], lw=1.2, linestyle="--", label="WT contour (pred)"),
    ]

    for row, orientation in enumerate(ORIENTATION_ORDER):
        idx = None if not indices else indices.get(orientation)
        loc = slice_index(pred, orientation, idx)  # type: ignore[arg-type]
        mri = mri_plane(image, orientation, loc)
        pred_sl = extract_slice(pred, orientation, loc)
        axes[row, 0].imshow(mri, cmap="gray", origin="lower")
        axes[row, 0].contour((pred_sl == 3).astype(float), levels=[0.5], colors=[CLASS_RGB[3]], linewidths=1.4)
        wt = np.isin(pred_sl, (1, 2, 3)).astype(float)
        axes[row, 0].contour(wt, levels=[0.5], colors=[CLASS_RGB[1]], linewidths=0.9, linestyles="--")
        axes[row, 0].set_title(f"{orientation}: MRI + pred contours")
        axes[row, 1].imshow(labels_to_rgb(pred_sl), origin="lower")
        axes[row, 1].set_title(f"{orientation}: predicted labels")
        if true is None:
            axes[row, 2].imshow(np.zeros((*mri.shape, 3)), origin="lower")
            axes[row, 2].text(
                0.5,
                0.5,
                "Ground truth not provided",
                ha="center",
                va="center",
                transform=axes[row, 2].transAxes,
                color="white",
                fontsize=8,
            )
            axes[row, 2].set_title(f"{orientation}: GT unavailable")
            axes[row, 3].imshow(np.zeros((*mri.shape, 3)), origin="lower")
            axes[row, 3].set_title(f"{orientation}: error N/A")
        else:
            true_sl = extract_slice(true, orientation, loc)
            axes[row, 2].imshow(labels_to_rgb(true_sl), origin="lower")
            axes[row, 2].set_title(f"{orientation}: ground-truth labels")
            err = error_rgb(pred_sl, true_sl, region="WT")
            axes[row, 3].imshow(err, origin="lower")
            axes[row, 3].set_title(f"{orientation}: WT error")
        for col in range(4):
            axes[row, col].set_xticks([])
            axes[row, col].set_yticks([])

    _add_class_legend(axes[0, 1], class_handles)
    axes[0, 0].legend(handles=contour_handles, loc="lower right", fontsize=7)
    if true is not None:
        axes[0, 3].legend(handles=error_handles, loc="lower right", fontsize=7)
    fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


def save_orthogonal_attribution(
    image: np.ndarray,
    cam: np.ndarray,
    path: Path | str,
    *,
    entropy: np.ndarray | None = None,
    indices: dict[str, int] | None = None,
    cam_label: str = "Grad-CAM (normalised, unitless)",
    title: str = DISCLAIMER,
) -> Path:
    plt, _Patch, _Line2D = _require_matplotlib()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    n_cols = 3 if entropy is not None else 2
    fig, axes = plt.subplots(3, n_cols, figsize=(3.4 * n_cols, 9.2))
    if n_cols == 2:
        axes = np.asarray(axes).reshape(3, 2)
    cam = np.asarray(cam, dtype=np.float64)
    for row, orientation in enumerate(ORIENTATION_ORDER):
        idx = None if not indices else indices.get(orientation)
        loc = slice_index(cam, orientation, idx)  # type: ignore[arg-type]
        mri = mri_plane(image, orientation, loc)
        cam_sl = extract_slice(cam, orientation, loc)
        axes[row, 0].imshow(mri, cmap="gray", origin="lower")
        heat = axes[row, 0].imshow(cam_sl, cmap="magma", origin="lower", alpha=0.55, vmin=0.0, vmax=1.0)
        axes[row, 0].set_title(f"{orientation}: MRI + {cam_label}")
        im_c = axes[row, 1].imshow(cam_sl, cmap="magma", origin="lower", vmin=0.0, vmax=1.0)
        axes[row, 1].set_title(f"{orientation}: {cam_label}")
        fig.colorbar(im_c, ax=axes[row, 1], fraction=0.046, pad=0.04)
        if entropy is not None:
            ent_sl = extract_slice(np.asarray(entropy), orientation, loc)
            im_e = axes[row, 2].imshow(ent_sl, cmap="viridis", origin="lower")
            axes[row, 2].set_title(f"{orientation}: predictive entropy (nats)")
            fig.colorbar(im_e, ax=axes[row, 2], fraction=0.046, pad=0.04)
        for col in range(n_cols):
            axes[row, col].set_xticks([])
            axes[row, col].set_yticks([])
        if row == 0:
            fig.colorbar(heat, ax=axes[row, 0], fraction=0.046, pad=0.04)
    fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path
