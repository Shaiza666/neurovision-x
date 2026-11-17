"""Validate uploaded multimodal NIfTI. Missing modalities are errors, not zeros."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from neurovisionx.data.io import IMAGE_MODALITIES, NIFTI_SUFFIXES, load_nifti
from neurovisionx.data.preprocess import remap_seg, zscore_channels

REQUIRED = IMAGE_MODALITIES  # t1, t1ce, t2, flair


class UploadValidationError(ValueError):
    """User-facing upload problem. Do not catch and replace with a dummy volume."""


def _is_nifti_name(name: str) -> bool:
    lower = name.lower()
    return any(lower.endswith(suffix) for suffix in NIFTI_SUFFIXES)


def write_named_bytes(directory: Path, filename: str, payload: bytes) -> Path:
    if not _is_nifti_name(filename):
        raise UploadValidationError(
            f"{filename!r} is not a .nii or .nii.gz file. Upload NIfTI modalities only."
        )
    directory.mkdir(parents=True, exist_ok=True)
    target = directory / Path(filename).name
    target.write_bytes(payload)
    return target


def validate_modality_paths(files: dict[str, Path | None]) -> dict[str, Path]:
    missing = [name for name in REQUIRED if not files.get(name)]
    if missing:
        raise UploadValidationError(
            "Missing required modalities: "
            + ", ".join(missing)
            + ". NeuroVision-X needs co-registered T1, T1ce, T2, and FLAIR. "
            "Empty channels are not synthesized."
        )
    resolved = {name: Path(files[name]) for name in REQUIRED}
    if files.get("seg"):
        resolved["seg"] = Path(files["seg"])
    return resolved


def load_validated_case(files: dict[str, Path]) -> dict[str, np.ndarray]:
    """Load stacked z-scored image (C,D,H,W) and optional seg. Shapes must match."""
    channels = []
    shapes = []
    for name in REQUIRED:
        volume = np.asarray(load_nifti(files[name]))
        if volume.ndim != 3:
            raise UploadValidationError(f"{name} must be a 3D NIfTI, got shape {volume.shape}")
        if not np.isfinite(volume).all():
            raise UploadValidationError(f"{name} contains non-finite voxels")
        channels.append(volume)
        shapes.append(volume.shape)
    if len(set(shapes)) != 1:
        detail = ", ".join(f"{n}={s}" for n, s in zip(REQUIRED, shapes))
        raise UploadValidationError(
            "Modalities are not co-registered (shapes differ): "
            + detail
            + ". Resample offline; this demo does not invent alignments."
        )
    image = zscore_channels(np.stack(channels, axis=0))
    out: dict[str, np.ndarray] = {"image": image, "shape": np.array(shapes[0])}
    if "seg" in files:
        seg = remap_seg(load_nifti(files["seg"]))
        if tuple(seg.shape) != tuple(shapes[0]):
            raise UploadValidationError(
                f"Segmentation shape {seg.shape} does not match image {shapes[0]}"
            )
        out["seg"] = seg
    return out
