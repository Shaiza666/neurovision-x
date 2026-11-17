"""Tiny NIfTI fixtures for unit tests. These are not BraTS and not research metrics."""

from __future__ import annotations

from pathlib import Path

import numpy as np


def write_nifti(path: Path, data: np.ndarray, affine: np.ndarray | None = None) -> Path:
    import nibabel as nib

    path.parent.mkdir(parents=True, exist_ok=True)
    if affine is None:
        affine = np.diag([1.0, 1.0, 1.0, 1.0])
    nib.save(nib.Nifti1Image(np.asanyarray(data), affine), str(path))
    return path


def brain_volume(shape: tuple[int, int, int] = (16, 16, 16), offset: float = 0.0) -> np.ndarray:
    volume = np.zeros(shape, dtype=np.float32)
    volume[2:-2, 2:-2, 2:-2] = 12.0 + offset
    interior = volume != 0
    zz, yy, xx = np.meshgrid(
        np.arange(shape[0]), np.arange(shape[1]), np.arange(shape[2]), indexing="ij"
    )
    volume[interior] += 0.05 * xx[interior].astype(np.float32)
    return volume


def tumor_seg(shape: tuple[int, int, int] = (16, 16, 16), legacy_et: bool = True) -> np.ndarray:
    seg = np.zeros(shape, dtype=np.int16)
    seg[5:11, 5:11, 5:11] = 1
    seg[6:10, 6:10, 6:10] = 2
    seg[7:9, 7:9, 7:9] = 4 if legacy_et else 3
    return seg


def write_patient(
    folder: Path,
    stem: str,
    *,
    style: str = "2021",
    shape: tuple[int, int, int] = (16, 16, 16),
    skip: tuple[str, ...] = (),
    affine: np.ndarray | None = None,
    corrupt: tuple[str, ...] = (),
) -> Path:
    folder.mkdir(parents=True, exist_ok=True)
    names_2021 = {
        "t1": f"{stem}_t1.nii.gz",
        "t1ce": f"{stem}_t1ce.nii.gz",
        "t2": f"{stem}_t2.nii.gz",
        "flair": f"{stem}_flair.nii.gz",
        "seg": f"{stem}_seg.nii.gz",
    }
    names_2023 = {
        "t1": f"{stem}-t1n.nii.gz",
        "t1ce": f"{stem}-t1c.nii.gz",
        "t2": f"{stem}-t2w.nii.gz",
        "flair": f"{stem}-t2f.nii.gz",
        "seg": f"{stem}-seg.nii.gz",
    }
    names = names_2023 if style == "2023" else names_2021
    for key, filename in names.items():
        if key in skip:
            continue
        path = folder / filename
        if key in corrupt:
            path.write_bytes(b"not a nifti file")
            continue
        if key == "seg":
            write_nifti(path, tumor_seg(shape), affine)
        else:
            offset = {"t1": 0.0, "t1ce": 1.0, "t2": 2.0, "flair": 3.0}[key]
            write_nifti(path, brain_volume(shape, offset), affine)
    return folder


def write_mini_dataset(
    root: Path,
    n_patients: int = 6,
    *,
    style: str = "2021",
    nested: bool = False,
) -> Path:
    for index in range(n_patients):
        stem = f"BraTS2021_{index:05d}" if style == "2021" else f"BraTS-GLI-{index:05d}-000"
        if nested:
            split_name = "training" if index < n_patients - 1 else "validation"
            folder = root / split_name / stem
        else:
            folder = root / stem
        write_patient(folder, stem, style=style)
    return root
