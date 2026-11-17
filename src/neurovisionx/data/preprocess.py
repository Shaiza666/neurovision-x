"""Intensity and crop helpers for real volumes.

Statistics must come from training patients when used in a reported run.
Random tensors in unit tests are not medical data.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from neurovisionx.data.io import IMAGE_MODALITIES
from neurovisionx.data.spatial import load_aligned_patient
from neurovisionx.evaluation.regions import remap_brats_labels


def zscore_nonzero(volume: np.ndarray) -> np.ndarray:
    """Z-score using only non-zero voxels (common BraTS brain-mask practice)."""
    data = np.asarray(volume, dtype=np.float32)
    mask = data != 0
    if not np.any(mask):
        raise ValueError("Volume is all zeros; refusing to synthesize intensities")
    mean = float(data[mask].mean())
    std = float(data[mask].std())
    if std == 0:
        raise ValueError("Non-zero voxels have zero variance; check the source file")
    out = np.zeros_like(data)
    out[mask] = (data[mask] - mean) / std
    return out


def zscore_channels(image: np.ndarray) -> np.ndarray:
    """``image`` is (C, D, H, W). Each channel is normalized independently."""
    return np.stack([zscore_nonzero(channel) for channel in image], axis=0)


def remap_seg(seg: np.ndarray) -> np.ndarray:
    return remap_brats_labels(seg).astype(np.int64)


def crop_or_pad(
    image: np.ndarray,
    seg: np.ndarray,
    patch_size: tuple[int, int, int],
    *,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Crop a (C, D, H, W) image and (D, H, W) seg to ``patch_size``.

    Random crop when ``rng`` is set (training). Center crop otherwise.
    Pads with zeros if a volume is smaller than the patch. Padding is not a
    synthetic tumor; it is empty background.
    """
    if image.ndim != 4:
        raise ValueError("image must be (C, D, H, W)")
    if seg.shape != image.shape[1:]:
        raise ValueError("seg shape must match image spatial shape")
    spatial = np.array(image.shape[1:], dtype=int)
    target = np.array(patch_size, dtype=int)
    pad_before = np.maximum((target - spatial + 1) // 2, 0)
    pad_after = np.maximum(target - spatial - pad_before, 0)
    if np.any(pad_before) or np.any(pad_after):
        image_pad = [(0, 0), *zip(pad_before.tolist(), pad_after.tolist())]
        image = np.pad(image, image_pad, mode="constant")
        seg = np.pad(seg, list(zip(pad_before.tolist(), pad_after.tolist())), mode="constant")
        spatial = np.array(image.shape[1:], dtype=int)

    starts = []
    for size, want in zip(spatial, target):
        extra = int(size - want)
        if extra <= 0:
            starts.append(0)
        elif rng is None:
            starts.append(extra // 2)
        else:
            starts.append(int(rng.integers(0, extra + 1)))
    slices = tuple(slice(start, start + want) for start, want in zip(starts, target))
    return image[(slice(None),) + slices], seg[slices]


def pad_to_min_shape(
    image: np.ndarray,
    seg: np.ndarray,
    patch_size: tuple[int, int, int],
) -> tuple[np.ndarray, np.ndarray]:
    """Pad only; do not crop. Used before grid / biased patch sampling."""
    if image.ndim != 4:
        raise ValueError("image must be (C, D, H, W)")
    spatial = np.array(image.shape[1:], dtype=int)
    target = np.array(patch_size, dtype=int)
    pad_before = np.maximum((target - spatial + 1) // 2, 0)
    pad_after = np.maximum(target - spatial - pad_before, 0)
    if not (np.any(pad_before) or np.any(pad_after)):
        return image, seg
    image_pad = [(0, 0), *zip(pad_before.tolist(), pad_after.tolist())]
    image = np.pad(image, image_pad, mode="constant")
    seg = np.pad(seg, list(zip(pad_before.tolist(), pad_after.tolist())), mode="constant")
    return image, seg


def stack_modalities(arrays: dict[str, np.ndarray], keys: tuple[str, ...] = IMAGE_MODALITIES) -> np.ndarray:
    missing = [key for key in keys if key not in arrays]
    if missing:
        raise KeyError(f"Missing modalities for stack: {missing}")
    return np.stack([arrays[key] for key in keys], axis=0)


def preprocess_patient_files(
    files: dict[str, Path],
    *,
    target_spacing_mm: tuple[float, float, float] = (1.0, 1.0, 1.0),
    crop_foreground: bool = True,
    intensity: str = "zscore_nonzero",
    margin: int = 2,
) -> dict[str, np.ndarray]:
    """Full NiBabel preprocess: RAS, spacing, crop, per-channel z-score."""
    arrays = load_aligned_patient(
        files,
        target_spacing_mm=target_spacing_mm,
        crop_foreground=crop_foreground,
        margin=margin,
    )
    if intensity == "zscore_nonzero":
        for key in IMAGE_MODALITIES:
            if key in arrays:
                arrays[key] = zscore_nonzero(arrays[key])
    elif intensity not in {"none", None}:
        raise ValueError(f"Unknown intensity mode {intensity!r}")
    return arrays
