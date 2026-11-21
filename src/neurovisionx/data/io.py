"""Locate and load BraTS-style NIfTI files from the tree that is actually present.

Filename tokens cover common 2021-style suffixes (``_t1``, ``_t1ce``, ``_t2``,
``_flair``) and 2023+ glioma suffixes (``-t1n``, ``-t1c``, ``-t2w``, ``-t2f``).
Missing files are reported; they are not synthesized.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from neurovisionx.data.errors import CorruptedNiftiError, MissingBraTSDataError

NIFTI_SUFFIXES = (".nii.gz", ".nii")
IMAGE_MODALITIES = ("t1", "t1ce", "t2", "flair")
ALL_KEYS = (*IMAGE_MODALITIES, "seg")

# More specific tokens first. ``t1`` must not steal ``t1ce`` / ``t1c``.
MODALITY_SUFFIXES: dict[str, tuple[str, ...]] = {
    "t1ce": ("t1ce", "t1c"),
    "t1": ("t1n", "t1"),
    "flair": ("flair", "t2f"),
    "t2": ("t2w", "t2"),
    "seg": ("seg",),
}


def _stem_without_nifti(name: str) -> str:
    lower = name.lower()
    for suffix in NIFTI_SUFFIXES:
        if lower.endswith(suffix):
            return lower[: -len(suffix)]
    return lower


def _token_match(stem: str, token: str) -> bool:
    return stem == token or stem.endswith(f"_{token}") or stem.endswith(f"-{token}")


def list_nifti(folder: Path) -> list[Path]:
    files: list[Path] = []
    if not folder.is_dir():
        return files
    for path in sorted(folder.iterdir()):
        lower = path.name.lower()
        if path.is_file() and lower.endswith(NIFTI_SUFFIXES):
            files.append(path)
    return files


def find_modality(folder: Path, modality: str) -> Path:
    tokens = MODALITY_SUFFIXES.get(modality)
    if not tokens:
        raise ValueError(f"Unknown modality {modality!r}")
    hits = [
        path
        for path in list_nifti(folder)
        if any(_token_match(_stem_without_nifti(path.name), token) for token in tokens)
    ]
    if not hits:
        raise MissingBraTSDataError(
            f"No {modality} NIfTI in {folder}. Official BraTS files are required."
        )
    if len(hits) > 1:
        names = ", ".join(path.name for path in hits)
        raise MissingBraTSDataError(f"Ambiguous {modality} files in {folder}: {names}")
    return hits[0]


def find_available_modalities(folder: Path) -> dict[str, Path]:
    """Return only the keys whose files exist in ``folder``."""
    found: dict[str, Path] = {}
    for name in ALL_KEYS:
        try:
            found[name] = find_modality(folder, name)
        except MissingBraTSDataError:
            continue
    return found


def find_available_in_files(files: list[Path]) -> dict[str, Path]:
    """Match known tokens among an explicit file list (flat layouts)."""
    found: dict[str, Path] = {}
    for name, tokens in MODALITY_SUFFIXES.items():
        hits = [
            path
            for path in files
            if any(_token_match(_stem_without_nifti(path.name), token) for token in tokens)
        ]
        if len(hits) == 1:
            found[name] = hits[0]
        elif len(hits) > 1:
            names = ", ".join(path.name for path in hits)
            raise MissingBraTSDataError(f"Ambiguous {name} files: {names}")
    return found


def infer_patient_id(filename: str) -> str | None:
    """Strip a trailing modality token from a NIfTI filename stem."""
    original_stem = Path(filename).name
    for suffix in NIFTI_SUFFIXES:
        if original_stem.lower().endswith(suffix):
            original_stem = original_stem[: -len(suffix)]
            break
    lower = original_stem.lower()
    for tokens in MODALITY_SUFFIXES.values():
        for token in tokens:
            for sep in ("_", "-"):
                suffix = f"{sep}{token}"
                if lower.endswith(suffix):
                    return original_stem[: -len(suffix)]
    return None


def infer_naming(files: dict[str, Path]) -> str:
    names = " ".join(path.name.lower() for path in files.values())
    has_2023 = any(token in names for token in ("-t1n", "-t1c", "-t2w", "-t2f", "t1n", "t2f"))
    has_2021 = any(token in names for token in ("_t1ce", "_flair", "_t1.", "t1ce", "flair"))
    if has_2023 and not has_2021:
        return "brats2023_glioma"
    if has_2021 and not has_2023:
        return "brats2021"
    if has_2023 and has_2021:
        return "mixed"
    return "unknown"


def resolve_patient_files(patient_dir: Path) -> dict[str, Path]:
    """Return paths for t1, t1ce, t2, flair, and seg."""
    if not patient_dir.is_dir():
        raise MissingBraTSDataError(f"Patient folder missing: {patient_dir}")
    return {name: find_modality(patient_dir, name) for name in ALL_KEYS}


def _require_nibabel() -> Any:
    try:
        import nibabel as nib
    except ImportError as exc:
        raise ImportError(
            'nibabel is required to read NIfTI. Install with: pip install -e ".[research]"'
        ) from exc
    return nib


def load_nifti_image(path: Path) -> Any:
    """Load a NiBabel image without fabricating voxels."""
    nib = _require_nibabel()
    try:
        return nib.load(str(path))
    except Exception as exc:  # nibabel raises several types for truncated gzip
        raise CorruptedNiftiError(f"Cannot read NIfTI: {path} ({exc})") from exc


@dataclass(frozen=True)
class NiftiInfo:
    path: str
    shape: tuple[int, ...]
    dtype: str
    zooms_mm: tuple[float, ...]
    axcodes: tuple[str, ...]
    affine: tuple[tuple[float, ...], ...]
    ndim: int


def nifti_info(path: Path) -> NiftiInfo:
    nib = _require_nibabel()
    image = load_nifti_image(path)
    shape = tuple(int(v) for v in image.shape)
    zooms = tuple(float(v) for v in image.header.get_zooms()[: min(3, len(shape))])
    affine = np.asarray(image.affine, dtype=np.float64)
    axcodes = tuple(str(code) for code in nib.aff2axcodes(affine))
    try:
        dataobj = image.dataobj
        dtype = str(getattr(dataobj, "dtype", image.header.get_data_dtype()))
    except Exception:
        dtype = str(image.header.get_data_dtype())
    return NiftiInfo(
        path=str(path),
        shape=shape,
        dtype=dtype,
        zooms_mm=zooms,
        axcodes=axcodes,
        affine=tuple(tuple(float(x) for x in row) for row in affine.tolist()),
        ndim=len(shape),
    )


def load_nifti(path: Path) -> np.ndarray:
    """Load a NIfTI array. Requires nibabel; does not invent voxels."""
    image = load_nifti_image(path)
    try:
        data = np.asanyarray(image.dataobj)
    except Exception as exc:
        raise CorruptedNiftiError(f"Cannot read voxel data: {path} ({exc})") from exc
    if data.size == 0:
        raise CorruptedNiftiError(f"Empty voxel array: {path}")
    return data


def save_nifti(
    path: Path,
    data: np.ndarray,
    *,
    affine: np.ndarray | None = None,
    reference: Path | None = None,
) -> Path:
    """Write an array as NIfTI. Does not invent voxels; ``data`` must already exist.

    Uses ``reference`` affine/header when provided so predictions stay in the
    same space as the loaded MRI. Identity affine is used only if neither is given.
    """
    nib = _require_nibabel()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    header = None
    if reference is not None:
        image = load_nifti_image(reference)
        affine = np.asarray(image.affine, dtype=np.float64)
        header = image.header.copy()
    if affine is None:
        affine = np.eye(4, dtype=np.float64)
    out = nib.Nifti1Image(np.asanyarray(data), affine, header=header)
    nib.save(out, str(path))
    return path
