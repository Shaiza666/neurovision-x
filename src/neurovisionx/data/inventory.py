"""Discover patient cases from the directory tree that is actually on disk.

No layout is assumed up front. The walker records patient folders, flat file
groups, official train/validation subfolders, available modalities, and the
filename convention that was observed.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path

from neurovisionx.data.errors import MissingBraTSDataError
from neurovisionx.data.io import (
    ALL_KEYS,
    IMAGE_MODALITIES,
    find_available_in_files,
    find_available_modalities,
    infer_naming,
    infer_patient_id,
    list_nifti,
)

SKIP_DIR_NAMES = {".git", "__pycache__", ".ipynb_checkpoints", ".venv"}
OFFICIAL_SUBDIRS = ("training", "train", "validation", "val", "valid", "test", "testing")


@dataclass(frozen=True)
class PatientCase:
    patient_id: str
    folder: str
    files: dict[str, str]
    missing: tuple[str, ...]
    naming: str
    source_subdir: str | None
    unlabeled: bool

    def path_map(self) -> dict[str, Path]:
        return {key: Path(value) for key, value in self.files.items()}

    def is_complete(self) -> bool:
        return not self.missing and not self.unlabeled


@dataclass
class DatasetInventory:
    raw_root: str
    layout: str
    n_cases: int
    n_complete: int
    n_unlabeled: int
    naming_counts: dict[str, int]
    official_subdirs: tuple[str, ...]
    cases: list[PatientCase] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def complete_ids(self) -> list[str]:
        return [case.patient_id for case in self.cases if case.is_complete()]

    def labeled_ids(self) -> list[str]:
        return [case.patient_id for case in self.cases if not case.unlabeled and "seg" in case.files]

    def by_id(self) -> dict[str, PatientCase]:
        return {case.patient_id: case for case in self.cases}

    def to_dict(self) -> dict:
        payload = asdict(self)
        payload["clinical_disclaimer"] = (
            "Research prototype. Not a clinically validated diagnostic tool. "
            "This inventory describes files on disk; it is not a clinical report."
        )
        return payload


def _source_subdir(raw_root: Path, folder: Path) -> str | None:
    try:
        relative = folder.relative_to(raw_root)
    except ValueError:
        return None
    parts = relative.parts
    if parts and parts[0].lower() in OFFICIAL_SUBDIRS:
        return parts[0]
    return None


def _walk_dirs(root: Path, max_depth: int) -> list[Path]:
    found = [root]
    if max_depth <= 0:
        return found
    try:
        children = sorted(root.iterdir())
    except OSError:
        return found
    for child in children:
        if not child.is_dir() or child.name in SKIP_DIR_NAMES or child.name.startswith("."):
            continue
        found.extend(_walk_dirs(child, max_depth - 1))
    return found


def _case_from_files(
    patient_id: str,
    folder: Path,
    files: dict[str, Path],
    raw_root: Path,
) -> PatientCase:
    missing = tuple(key for key in ALL_KEYS if key not in files)
    unlabeled = "seg" not in files
    return PatientCase(
        patient_id=patient_id,
        folder=str(folder),
        files={key: str(path) for key, path in files.items()},
        missing=missing,
        naming=infer_naming(files),
        source_subdir=_source_subdir(raw_root, folder),
        unlabeled=unlabeled,
    )


def _cases_from_folder(folder: Path, raw_root: Path) -> list[PatientCase]:
    niftis = list_nifti(folder)
    if not niftis:
        return []
    available = find_available_modalities(folder)
    if available:
        try:
            relative = folder.relative_to(raw_root).as_posix()
        except ValueError:
            relative = folder.name
        patient_id = relative if relative not in {".", ""} else folder.name
        return [_case_from_files(patient_id, folder, available, raw_root)]

    grouped: dict[str, list[Path]] = {}
    unrecognized: list[str] = []
    for path in niftis:
        patient_id = infer_patient_id(path.name)
        if patient_id is None:
            unrecognized.append(path.name)
            continue
        grouped.setdefault(patient_id, []).append(path)
    cases = []
    for patient_id, group in sorted(grouped.items()):
        files = find_available_in_files(group)
        if files:
            cases.append(_case_from_files(patient_id, folder, files, raw_root))
    return cases


def discover_inventory(raw_root: Path, max_depth: int = 6) -> DatasetInventory:
    """Inspect ``raw_root`` and describe whatever BraTS-like cases exist."""
    raw_root = Path(raw_root)
    if not raw_root.exists():
        raise MissingBraTSDataError(
            f"No BraTS volumes under {raw_root}. Register at the official "
            "BraTS / Synapse distribution, accept the license, and place "
            "patient folders here. Synthetic MRI is not used for metrics."
        )

    seen_ids: set[str] = set()
    cases: list[PatientCase] = []
    notes: list[str] = []
    for folder in _walk_dirs(raw_root, max_depth):
        for case in _cases_from_folder(folder, raw_root):
            if case.patient_id in seen_ids:
                notes.append(f"duplicate patient_id skipped: {case.patient_id} ({case.folder})")
                continue
            seen_ids.add(case.patient_id)
            cases.append(case)

    if not cases:
        raise MissingBraTSDataError(
            f"No BraTS-like NIfTI cases under {raw_root}. "
            "Place official patient folders or challenge archives here. "
            "Synthetic MRI is not used for metrics."
        )

    official = tuple(
        sorted({case.source_subdir for case in cases if case.source_subdir is not None})
    )
    naming_counts: dict[str, int] = {}
    for case in cases:
        naming_counts[case.naming] = naming_counts.get(case.naming, 0) + 1
    n_complete = sum(1 for case in cases if case.is_complete())
    n_unlabeled = sum(1 for case in cases if case.unlabeled)
    if official:
        layout = "official_subfolders"
    elif any(Path(case.folder).resolve() == raw_root.resolve() for case in cases):
        layout = "flat_or_root"
    else:
        layout = "patient_folders"
    if n_complete == 0:
        notes.append("No case has all four MRI modalities plus a seg map.")
    missing_any = sum(1 for case in cases if case.missing)
    if missing_any:
        notes.append(f"{missing_any} case(s) are missing at least one expected file.")

    return DatasetInventory(
        raw_root=str(raw_root),
        layout=layout,
        n_cases=len(cases),
        n_complete=n_complete,
        n_unlabeled=n_unlabeled,
        naming_counts=naming_counts,
        official_subdirs=official,
        cases=sorted(cases, key=lambda case: case.patient_id),
        notes=notes,
    )


def discover_patient_cases(raw_root: Path, max_depth: int = 6) -> list[PatientCase]:
    return discover_inventory(raw_root, max_depth=max_depth).cases


def image_keys_present(case: PatientCase) -> tuple[str, ...]:
    return tuple(key for key in IMAGE_MODALITIES if key in case.files)
