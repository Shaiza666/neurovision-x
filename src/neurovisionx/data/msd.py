"""Medical Segmentation Decathlon Task01_BrainTumour → this repo's BraTS layout.

Task01 is the Decathlon brain-tumour task (Simpson et al., 2019; Antonelli et
al., 2022), derived from BraTS 2016/2017. License: CC-BY-SA 4.0. It is **not**
the official BraTS 2021+ Synapse release and must not be reported as one.

Decathlon ``dataset.json`` (confirmed at convert time, not assumed silently):

- Modalities, channel axis of the 4D image: FLAIR, T1w, T1gd, T2w.
- Labels: 0 background, 1 edema, 2 non-enhancing tumor, 3 enhancing tumour.

This repository scores nested regions with BraTS integers
(``evaluation/regions.py``):

- 1 NCR (necrotic / non-enhancing core)
- 2 ED (edema)
- 3 ET (enhancing tumor)

The convert step therefore remaps MSD labels ``1 → 2``, ``2 → 1``, ``3 → 3``.
Channels are written as separate NIfTI files whose names use this repo's
2021-style tokens (``_t1``, ``_t1ce``, ``_t2``, ``_flair``, ``_seg``). The
filename convention is only a loader contract. The voxels remain MSD.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from neurovisionx.data.splits import PatientSplit, make_split

# Published MONAI mirror. The convert script records the bytes it actually
# downloaded; this constant is the URL we request.
MSD_TASK01_URL = "https://msd-for-monai.s3-us-west-2.amazonaws.com/Task01_BrainTumour.tar"
MSD_TASK01_MONAI_MD5 = "240a19d752f0d9e9101544901065d872"

# MSD label integer → BraTS integer used by nested WT/TC/ET scoring.
MSD_TO_BRATS_LABEL = {0: 0, 1: 2, 2: 1, 3: 3}

_MODALITY_ALIASES = {
    "flair": "flair",
    "t2f": "flair",
    "t1w": "t1",
    "t1": "t1",
    "t1gd": "t1ce",
    "t1c": "t1ce",
    "t1ce": "t1ce",
    "t2w": "t2",
    "t2": "t2",
}


class MSDSchemaError(ValueError):
    """dataset.json does not match the Task01 mapping this repo implements."""


def _norm_token(value: Any) -> str:
    return "".join(ch for ch in str(value).lower() if ch.isalnum())


def modality_index(modality: dict[Any, Any]) -> dict[str, int]:
    """Map repo keys ``t1`` / ``t1ce`` / ``t2`` / ``flair`` to 4D channel indexes.

    Indexes come from ``dataset.json`` so a future file with a different order
    is rejected or mapped explicitly instead of being stacked as BraTS order.
    """
    if not isinstance(modality, dict) or not modality:
        raise MSDSchemaError("dataset.json is missing the modality map")
    found: dict[str, int] = {}
    for key, name in modality.items():
        token = _norm_token(name)
        if token not in _MODALITY_ALIASES:
            raise MSDSchemaError(f"Unrecognised MSD modality name {name!r}")
        repo_key = _MODALITY_ALIASES[token]
        if repo_key in found:
            raise MSDSchemaError(f"Duplicate modality for {repo_key}: {name!r}")
        found[repo_key] = int(key)
    expected = {"t1", "t1ce", "t2", "flair"}
    if set(found) != expected:
        raise MSDSchemaError(f"Modality map resolved to {found}, expected {expected}")
    return found


def assert_task01_labels(labels: dict[Any, Any]) -> None:
    """Require the Decathlon edema / non-enhancing / enhancing coding.

    Fails closed. A different coding must not be remapped with this table.
    """
    if not isinstance(labels, dict):
        raise MSDSchemaError("dataset.json is missing labels")
    by_id = {int(key): str(value).lower().replace("—", "-").replace("–", "-") for key, value in labels.items()}
    background = by_id.get(0, "")
    edema = by_id.get(1, "")
    nonenhancing = by_id.get(2, "")
    enhancing = by_id.get(3, "")
    if "background" not in background:
        raise MSDSchemaError(f"Label 0 is {by_id.get(0)!r}, expected background")
    if "edema" not in edema and "oedema" not in edema:
        raise MSDSchemaError(f"Label 1 is {by_id.get(1)!r}, expected edema")
    non_tokens = ("non-enhancing", "nonenhancing", "necrotic", "non-enhancing tumor", "non-enhancing tumour")
    if not any(token in nonenhancing for token in non_tokens):
        raise MSDSchemaError(f"Label 2 is {by_id.get(2)!r}, expected non-enhancing tumor")
    if "enhancing" not in enhancing or "non-enhancing" in enhancing or "nonenhancing" in enhancing:
        raise MSDSchemaError(f"Label 3 is {by_id.get(3)!r}, expected enhancing tumour")


def remap_msd_labels(seg: np.ndarray) -> np.ndarray:
    """Map MSD Task01 integers onto this repo's BraTS class ids.

    Unexpected values raise. They are not collapsed into background.
    """
    array = np.asarray(seg)
    if not np.issubdtype(array.dtype, np.integer):
        rounded = np.rint(array)
        if not np.allclose(array, rounded):
            raise MSDSchemaError("Segmentation is not integer-valued; refusing to remap")
        array = rounded
    array = np.asarray(array, dtype=np.int16)
    unexpected = sorted({int(value) for value in np.unique(array)} - set(MSD_TO_BRATS_LABEL))
    if unexpected:
        raise MSDSchemaError(f"Unexpected MSD label values {unexpected}")
    out = np.zeros(array.shape, dtype=np.int16)
    out[array == 1] = np.int16(2)  # edema → ED
    out[array == 2] = np.int16(1)  # non-enhancing → NCR
    out[array == 3] = np.int16(3)  # enhancing → ET
    return out


def split_msd_image(image: np.ndarray, index: dict[str, int]) -> dict[str, np.ndarray]:
    """Split a 4D Decathlon volume into ``t1`` / ``t1ce`` / ``t2`` / ``flair``."""
    data = np.asarray(image)
    if data.ndim != 4:
        raise MSDSchemaError(f"Expected a 4D image, got shape {data.shape}")
    if data.shape[-1] == 4:
        channels = {key: np.asarray(data[..., channel], dtype=np.float32) for key, channel in index.items()}
    elif data.shape[0] == 4:
        channels = {key: np.asarray(data[channel], dtype=np.float32) for key, channel in index.items()}
    else:
        raise MSDSchemaError(f"Could not find 4 modality channels in shape {data.shape}")
    for key, volume in channels.items():
        if volume.ndim != 3:
            raise MSDSchemaError(f"{key} channel is not 3D: {volume.shape}")
        if not np.isfinite(volume).all():
            raise MSDSchemaError(f"{key} channel contains NaN or Inf")
    return channels


def fit_to_grid(
    image: np.ndarray,
    seg: np.ndarray,
    grid: tuple[int, int, int],
    *,
    center: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, dict[str, dict[str, float | int | None]]]:
    """Crop or pad ``image`` (C, D, H, W) and ``seg`` (D, H, W) to ``grid``.

    The window is centered on ``center`` (voxel indices) and clamped so a
    volume larger than ``grid`` keeps a full window. Smaller volumes are
    padded with zeros. Padding is empty background, not a synthetic tumor.
    Label retention is counted so a crop that clips ET is visible.
    """
    if image.ndim != 4:
        raise ValueError("image must be (C, D, H, W)")
    if seg.shape != image.shape[1:]:
        raise ValueError("seg shape must match image spatial shape")
    target = np.array([int(v) for v in grid], dtype=int)
    spatial = np.array(seg.shape, dtype=int)
    if center is None:
        coords = np.argwhere(seg > 0)
        if coords.size == 0:
            coords = np.argwhere(np.any(image != 0, axis=0))
        center = coords.mean(axis=0) if coords.size else spatial / 2.0
    center = np.asarray(center, dtype=np.float64)
    start = np.zeros(3, dtype=int)
    for axis in range(3):
        if spatial[axis] <= target[axis]:
            # Negative start pads before the volume so the content stays centered.
            start[axis] = -int((target[axis] - spatial[axis]) // 2)
        else:
            proposed = int(np.rint(center[axis] - target[axis] / 2.0))
            start[axis] = int(np.clip(proposed, 0, spatial[axis] - target[axis]))

    src_slices: list[slice] = []
    dst_slices: list[slice] = []
    for axis in range(3):
        src_lo = int(max(start[axis], 0))
        src_hi = int(min(start[axis] + target[axis], spatial[axis]))
        dst_lo = int(src_lo - start[axis])
        dst_hi = int(dst_lo + (src_hi - src_lo))
        src_slices.append(slice(src_lo, src_hi))
        dst_slices.append(slice(dst_lo, dst_hi))

    out_image = np.zeros((image.shape[0], *target.tolist()), dtype=np.float32)
    out_seg = np.zeros(target.tolist(), dtype=np.int16)
    out_image[(slice(None), *dst_slices)] = image[(slice(None), *src_slices)]
    out_seg[tuple(dst_slices)] = seg[tuple(src_slices)]

    retained: dict[str, dict[str, float | int | None]] = {}
    for label, name in ((1, "ncr"), (2, "ed"), (3, "et")):
        before = int((seg == label).sum())
        after = int((out_seg == label).sum())
        retained[name] = {
            "label": label,
            "before": before,
            "after": after,
            "fraction": (after / before) if before else None,
        }
    return out_image, out_seg, retained


def subset_split(split: PatientSplit, limits: dict[str, int]) -> PatientSplit:
    """Take a deterministic prefix of an existing patient-level split.

    The prefix follows the seeded split order. It does not reshuffle, so the
    subset cannot move a test id into train.
    """
    train = tuple(split.train[: int(limits["train"])])
    val = tuple(split.val[: int(limits["val"])])
    test = tuple(split.test[: int(limits["test"])])
    if min(len(train), len(val), len(test)) < 1:
        raise ValueError(
            f"Subset limits {limits} emptied a split "
            f"(available train/val/test {len(split.train)}/{len(split.val)}/{len(split.test)})"
        )
    return make_split(train, val, test, split.seed)


def load_dataset_json(path: Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise MSDSchemaError(f"{path} is not a JSON object")
    return payload


def write_brats_style_case(
    image_path: Path,
    label_path: Path,
    dest_root: Path,
    index: dict[str, int],
) -> dict:
    """Write one MSD case as five BraTS-token NIfTI files. Idempotent."""
    from neurovisionx.data.io import load_nifti, load_nifti_image, save_nifti

    patient_id = patient_id_from_image_name(image_path.name)
    folder = dest_root / patient_id
    expected = [folder / f"{patient_id}_{token}.nii.gz" for token in ("t1", "t1ce", "t2", "flair", "seg")]
    if all(path.is_file() and path.stat().st_size > 0 for path in expected):
        return {"patient_id": patient_id, "status": "skipped_existing", "folder": str(folder)}

    image = load_nifti_image(image_path)
    affine = np.asarray(image.affine, dtype=np.float64)
    channels = split_msd_image(np.asanyarray(image.dataobj), index)
    seg = remap_msd_labels(load_nifti(label_path))
    spatial = next(iter(channels.values())).shape
    if tuple(seg.shape) != tuple(spatial):
        raise MSDSchemaError(f"{patient_id} label shape {seg.shape} != image {spatial}")
    folder.mkdir(parents=True, exist_ok=True)
    for key in ("t1", "t1ce", "t2", "flair"):
        save_nifti(folder / f"{patient_id}_{key}.nii.gz", channels[key], affine=affine)
    save_nifti(folder / f"{patient_id}_seg.nii.gz", seg, affine=affine)
    counts = {str(int(value)): int((seg == value).sum()) for value in (0, 1, 2, 3)}
    return {
        "patient_id": patient_id,
        "status": "converted",
        "folder": str(folder),
        "shape": list(spatial),
        "label_counts_brats_ids": counts,
    }


def patient_id_from_image_name(name: str) -> str:
    stem = Path(name).name
    lower = stem.lower()
    for suffix in (".nii.gz", ".nii"):
        if lower.endswith(suffix):
            stem = stem[: -len(suffix)]
            break
    return stem
