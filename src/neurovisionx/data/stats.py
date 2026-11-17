"""Dataset statistics from loaded volumes.

Voxel counts are measured on arrays that were actually read. Empty placeholders
are not written when data is missing.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from neurovisionx.data.inventory import PatientCase
from neurovisionx.data.io import IMAGE_MODALITIES, load_nifti
from neurovisionx.evaluation.regions import nested_masks, remap_brats_labels

CLASS_NAMES = {0: "background", 1: "ncr", 2: "ed", 3: "et"}


@dataclass
class PatientStats:
    patient_id: str
    shape: tuple[int, ...]
    class_voxels: dict[str, int]
    region_voxels: dict[str, int]
    modality_nonzero: dict[str, int]


def _count_classes(seg: np.ndarray) -> dict[str, int]:
    remapped = remap_brats_labels(seg)
    unique, counts = np.unique(remapped, return_counts=True)
    out = {name: 0 for name in CLASS_NAMES.values()}
    for value, count in zip(unique, counts):
        name = CLASS_NAMES.get(int(value))
        if name is None:
            out[f"other_{int(value)}"] = int(count)
        else:
            out[name] = int(count)
    return out


def _count_regions(seg: np.ndarray) -> dict[str, int]:
    masks = nested_masks(seg)
    return {name: int(mask.sum()) for name, mask in masks.items()}


def patient_stats(case: PatientCase) -> PatientStats:
    files = case.path_map()
    if "seg" not in files:
        raise ValueError(f"{case.patient_id} has no segmentation; stats not computed")
    seg = load_nifti(files["seg"])
    modality_nonzero = {}
    shape = tuple(int(v) for v in seg.shape)
    for key in IMAGE_MODALITIES:
        if key not in files:
            continue
        data = load_nifti(files[key])
        modality_nonzero[key] = int(np.count_nonzero(data))
        if data.shape != seg.shape:
            modality_nonzero[f"{key}_shape"] = int(np.prod(data.shape))
    return PatientStats(
        patient_id=case.patient_id,
        shape=shape,
        class_voxels=_count_classes(seg),
        region_voxels=_count_regions(seg),
        modality_nonzero=modality_nonzero,
    )


def aggregate_stats(rows: list[PatientStats]) -> dict[str, Any]:
    class_totals: Counter[str] = Counter()
    region_totals: Counter[str] = Counter()
    shapes: Counter[tuple[int, ...]] = Counter()
    for row in rows:
        class_totals.update(row.class_voxels)
        region_totals.update(row.region_voxels)
        shapes[row.shape] += 1
    n_voxels = sum(class_totals.values())
    class_fractions = {
        name: (count / n_voxels if n_voxels else 0.0) for name, count in class_totals.items()
    }
    return {
        "n_patients": len(rows),
        "class_voxels": dict(class_totals),
        "class_fractions": class_fractions,
        "region_voxels": dict(region_totals),
        "shapes": {"x".join(str(v) for v in shape): n for shape, n in shapes.items()},
        "patients": [
            {
                "patient_id": row.patient_id,
                "shape": list(row.shape),
                "class_voxels": row.class_voxels,
                "region_voxels": row.region_voxels,
                "modality_nonzero": row.modality_nonzero,
            }
            for row in rows
        ],
        "clinical_disclaimer": (
            "Research prototype. Not a clinically validated diagnostic tool. "
            "Counts are from files that were loaded; they are not a clinical report."
        ),
    }


def write_json(payload: dict[str, Any], path: Path) -> Path:
    import json

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path
