"""BraTS-style volume dataset.

Real NIfTI files are required. This module refuses to synthesize medical
volumes for reported metrics. The on-disk tree is discovered, not assumed.
"""

from __future__ import annotations

from pathlib import Path

from neurovisionx.data.errors import MissingBraTSDataError
from neurovisionx.data.inventory import PatientCase, discover_inventory, discover_patient_cases
from neurovisionx.data.io import list_nifti, resolve_patient_files

MODALITY_FILES = ("t1", "t1ce", "t2", "flair")


def _patient_dirs(raw_root: Path) -> list[Path]:
    if not raw_root.exists():
        return []
    return sorted(
        path
        for path in raw_root.iterdir()
        if path.is_dir() and not path.name.startswith(".")
    )


def require_brats_root(raw_root: Path) -> Path:
    raw_root = Path(raw_root)
    try:
        inventory = discover_inventory(raw_root)
    except MissingBraTSDataError:
        patients = _patient_dirs(raw_root)
        nifti = [path for patient in patients for path in list_nifti(patient)]
        if not patients or not nifti:
            raise
        return raw_root
    if inventory.n_cases == 0:
        raise MissingBraTSDataError(
            f"No BraTS volumes under {raw_root}. Register at the official "
            "BraTS / Synapse distribution, accept the license, and place "
            "patient folders here. Synthetic MRI is not used for metrics."
        )
    return raw_root


def list_patient_ids(raw_root: Path) -> list[str]:
    require_brats_root(raw_root)
    return [case.patient_id for case in discover_patient_cases(raw_root)]


def list_complete_cases(raw_root: Path) -> list[PatientCase]:
    inventory = discover_inventory(raw_root)
    return [case for case in inventory.cases if case.is_complete()]


def validate_patient_folder(patient_dir: Path) -> dict[str, Path]:
    """Require all four modalities and a seg map. Used before training."""
    return resolve_patient_files(patient_dir)
