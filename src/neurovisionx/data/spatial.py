"""Reproducible spatial preprocessing with NiBabel.

Established practice (nnU-Net / BraTS): reorient to RAS, resample to a common
1 mm grid when needed, crop the non-zero brain. This is not a novel method.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from neurovisionx.data.errors import CorruptedNiftiError
from neurovisionx.data.io import IMAGE_MODALITIES, load_nifti_image
from neurovisionx.evaluation.regions import remap_brats_labels


def _require_nibabel() -> Any:
    try:
        import nibabel as nib
    except ImportError as exc:
        raise ImportError(
            'nibabel is required for spatial preprocessing. pip install -e ".[research]"'
        ) from exc
    return nib


@dataclass(frozen=True)
class SpatialVolume:
    data: np.ndarray
    affine: np.ndarray
    zooms_mm: tuple[float, float, float]
    axcodes: tuple[str, ...]


def canonical_ras(path: Path) -> Any:
    """Load a NIfTI and reorient to the closest RAS canonical form."""
    nib = _require_nibabel()
    image = load_nifti_image(path)
    return nib.as_closest_canonical(image)


def _zooms3(image: Any) -> tuple[float, float, float]:
    zooms = tuple(float(v) for v in image.header.get_zooms()[:3])
    if len(zooms) != 3:
        raise CorruptedNiftiError(f"Expected 3D zooms, got {zooms}")
    return zooms  # type: ignore[return-value]


def resample_to_reference(image: Any, reference: Any, *, is_label: bool) -> Any:
    try:
        from nibabel.processing import resample_from_to
    except ImportError as exc:
        raise ImportError("nibabel.processing requires scipy") from exc
    order = 0 if is_label else 3
    return resample_from_to(image, (reference.shape, reference.affine), order=order)


def resample_to_spacing(image: Any, spacing_mm: tuple[float, float, float], *, is_label: bool) -> Any:
    try:
        from nibabel.processing import resample_to_output
    except ImportError as orig:
        zooms = _zooms3(image)
        if np.allclose(zooms, spacing_mm, atol=1e-3):
            return image
        raise ImportError("scipy/nibabel.processing is required to resample spacing") from orig
    order = 0 if is_label else 3
    zooms = _zooms3(image)
    if np.allclose(zooms, spacing_mm, atol=0.05):
        return image
    return resample_to_output(image, voxel_sizes=spacing_mm, order=order)


def spatial_volume(image: Any) -> SpatialVolume:
    nib = _require_nibabel()
    data = np.asanyarray(image.dataobj)
    affine = np.asarray(image.affine, dtype=np.float64)
    return SpatialVolume(
        data=data,
        affine=affine,
        zooms_mm=_zooms3(image),
        axcodes=tuple(str(code) for code in nib.aff2axcodes(affine)),
    )


def foreground_bbox(*volumes: np.ndarray, margin: int = 2) -> tuple[slice, slice, slice]:
    mask = np.zeros(volumes[0].shape, dtype=bool)
    for volume in volumes:
        mask |= np.asarray(volume) != 0
    coords = np.argwhere(mask)
    if coords.size == 0:
        raise ValueError("Foreground crop requested but all volumes are zero")
    mins = np.maximum(coords.min(axis=0) - margin, 0)
    maxs = np.minimum(coords.max(axis=0) + 1 + margin, mask.shape)
    return tuple(slice(int(lo), int(hi)) for lo, hi in zip(mins, maxs))  # type: ignore[return-value]


def crop_with_bbox(volume: np.ndarray, bbox: tuple[slice, slice, slice]) -> np.ndarray:
    return volume[bbox]


def load_aligned_patient(
    files: dict[str, Path],
    *,
    target_spacing_mm: tuple[float, float, float] = (1.0, 1.0, 1.0),
    crop_foreground: bool = True,
    margin: int = 2,
) -> dict[str, np.ndarray]:
    """RAS-align, match spacing/grid, optional foreground crop.

    Returns float32 channels ``t1``/``t1ce``/``t2``/``flair`` and int64 ``seg``
    when those files are present.
    """
    if not files:
        raise ValueError("No files to load")
    reference_key = next((key for key in ("t1ce", "t1", "flair", "t2") if key in files), next(iter(files)))
    aligned: dict[str, Any] = {}
    for key, path in files.items():
        image = canonical_ras(path)
        image = resample_to_spacing(image, target_spacing_mm, is_label=(key == "seg"))
        aligned[key] = image

    reference = aligned[reference_key]
    for key, image in list(aligned.items()):
        if key == reference_key:
            continue
        same_shape = tuple(image.shape) == tuple(reference.shape)
        same_affine = np.allclose(image.affine, reference.affine, atol=1e-3)
        if not (same_shape and same_affine):
            aligned[key] = resample_to_reference(image, reference, is_label=(key == "seg"))

    arrays = {key: np.asanyarray(image.dataobj) for key, image in aligned.items()}
    if crop_foreground:
        image_arrays = [arrays[key] for key in IMAGE_MODALITIES if key in arrays]
        if image_arrays:
            bbox = foreground_bbox(*image_arrays, margin=margin)
            arrays = {key: crop_with_bbox(value, bbox) for key, value in arrays.items()}
    if "seg" in arrays:
        arrays["seg"] = remap_brats_labels(arrays["seg"]).astype(np.int64)
    for key in IMAGE_MODALITIES:
        if key in arrays:
            arrays[key] = np.asarray(arrays[key], dtype=np.float32)
    return arrays
