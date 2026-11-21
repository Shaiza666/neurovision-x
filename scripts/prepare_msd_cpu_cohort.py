"""Build the reduced 2 mm / 96³ cohort from converted MSD patient folders.

The cohort is a prefix of data/splits/msd_task01_patients.json (seed 2026).
Cases are written under training/, validation/, and test/ so the existing
loader keeps those patients in the same splits instead of shuffling again.

World affine on the cohort is a 2 mm diagonal. It is not the original scanner
pose. HD95 on this tree is in voxels of that grid.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from neurovisionx.data.io import save_nifti
from neurovisionx.data.msd import fit_to_grid, subset_split
from neurovisionx.data.preprocess import stack_modalities
from neurovisionx.data.spatial import load_aligned_patient
from neurovisionx.data.splits import load_split, save_split


def _case_files(patient_dir: Path) -> dict[str, Path]:
    from neurovisionx.data.io import resolve_patient_files

    return resolve_patient_files(patient_dir)


def _write_case(folder: Path, patient_id: str, image: np.ndarray, seg: np.ndarray, spacing: tuple[float, float, float]) -> None:
    affine = np.diag([spacing[0], spacing[1], spacing[2], 1.0]).astype(np.float64)
    folder.mkdir(parents=True, exist_ok=True)
    channels = {"t1": 0, "t1ce": 1, "t2": 2, "flair": 3}
    for name, index in channels.items():
        save_nifti(folder / f"{patient_id}_{name}.nii.gz", image[index], affine=affine)
    save_nifti(folder / f"{patient_id}_seg.nii.gz", seg.astype(np.int16), affine=affine)


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare the reduced MSD CPU cohort.")
    parser.add_argument("--source", type=Path, default=ROOT / "data" / "raw" / "msd_task01")
    parser.add_argument("--split", type=Path, default=ROOT / "data" / "splits" / "msd_task01_patients.json")
    parser.add_argument("--dest", type=Path, default=ROOT / "data" / "raw" / "msd_task01_cpu")
    parser.add_argument("--protocol-out", type=Path, default=ROOT / "docs" / "msd_cpu_protocol.json")
    parser.add_argument("--spacing", type=float, nargs=3, default=(2.0, 2.0, 2.0))
    parser.add_argument("--grid", type=int, nargs=3, default=(96, 96, 96))
    parser.add_argument("--train", type=int, default=48)
    parser.add_argument("--val", type=int, default=8)
    parser.add_argument("--test", type=int, default=12)
    args = parser.parse_args()

    full = load_split(args.split)
    subset = subset_split(full, {"train": args.train, "val": args.val, "test": args.test})
    spacing = tuple(float(v) for v in args.spacing)
    grid = tuple(int(v) for v in args.grid)
    groups = {
        "training": subset.train,
        "validation": subset.val,
        "test": subset.test,
    }
    records = []
    for split_name, ids in groups.items():
        for patient_id in ids:
            source = args.source / patient_id
            if not source.is_dir():
                raise SystemExit(f"Missing converted patient folder: {source}")
            arrays = load_aligned_patient(
                _case_files(source),
                target_spacing_mm=spacing,
                crop_foreground=True,
                margin=2,
            )
            image = stack_modalities(arrays).astype(np.float32)
            seg = np.asarray(arrays["seg"], dtype=np.int16)
            coords = np.argwhere(seg > 0)
            center = coords.mean(axis=0) if coords.size else None
            fitted, fitted_seg, retained = fit_to_grid(image, seg, grid, center=center)
            dest = args.dest / split_name / patient_id
            _write_case(dest, patient_id, fitted, fitted_seg, spacing)
            records.append(
                {
                    "patient_id": patient_id,
                    "split": split_name,
                    "source_shape": list(image.shape[1:]),
                    "grid": list(grid),
                    "retention": retained,
                }
            )
            print(f"{split_name}/{patient_id} source={image.shape[1:]} et_keep={retained['et']['fraction']}")

    def _mean_fraction(name: str) -> float | None:
        values = [
            row["retention"][name]["fraction"]
            for row in records
            if row["retention"][name]["fraction"] is not None
        ]
        if not values:
            return None
        return float(sum(values) / len(values))

    protocol = {
        "dataset": "MSD Task01_BrainTumour (CC-BY-SA 4.0), not official BraTS 2021+",
        "source_split": str(args.split),
        "seed": full.seed,
        "full_split_counts": {
            "train": len(full.train),
            "val": len(full.val),
            "test": len(full.test),
        },
        "subset_counts": {"train": len(subset.train), "val": len(subset.val), "test": len(subset.test)},
        "subset_rule": "prefix of the seeded split lists; not a new shuffle",
        "train_ids": list(subset.train),
        "val_ids": list(subset.val),
        "test_ids": list(subset.test),
        "spacing_mm": list(spacing),
        "grid": list(grid),
        "affine": "diag(spacing); not the original scanner pose",
        "crop": "RAS + resample to spacing + foreground crop (margin 2) + center on tumor (or foreground) into grid",
        "mean_label_voxel_fraction_retained": {
            "ncr": _mean_fraction("ncr"),
            "ed": _mean_fraction("ed"),
            "et": _mean_fraction("et"),
        },
        "patients": records,
        "cohort_root": str(args.dest),
        "clinical_disclaimer": "Research prototype. Not a clinically validated diagnostic tool.",
    }
    args.protocol_out.parent.mkdir(parents=True, exist_ok=True)
    args.protocol_out.write_text(json.dumps(protocol, indent=2), encoding="utf-8")
    from neurovisionx.data.splits import make_split

    stored = make_split(
        [f"training/{pid}" for pid in subset.train],
        [f"validation/{pid}" for pid in subset.val],
        [f"test/{pid}" for pid in subset.test],
        full.seed,
    )
    save_split(stored, ROOT / "data" / "splits" / "msd_task01_cpu_patients.json")
    print(f"Wrote {args.protocol_out}")
    print(
        "Mean fraction retained "
        f"NCR={protocol['mean_label_voxel_fraction_retained']['ncr']} "
        f"ED={protocol['mean_label_voxel_fraction_retained']['ed']} "
        f"ET={protocol['mean_label_voxel_fraction_retained']['et']}"
    )


if __name__ == "__main__":
    main()
