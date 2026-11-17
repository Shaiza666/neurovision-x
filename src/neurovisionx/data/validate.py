"""Validate MRI volumes and labels that were found on disk.

Checks are computed from NiBabel headers and arrays. Failures are recorded;
they are not repaired with synthetic voxels.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from neurovisionx.data.errors import CorruptedNiftiError, DatasetValidationError
from neurovisionx.data.inventory import PatientCase
from neurovisionx.data.io import IMAGE_MODALITIES, NiftiInfo, load_nifti, nifti_info
from neurovisionx.evaluation.regions import remap_brats_labels

ALLOWED_LABELS = frozenset({0, 1, 2, 3, 4})


@dataclass
class FileIssue:
    key: str
    path: str
    code: str
    message: str


@dataclass
class PatientValidation:
    patient_id: str
    ok: bool
    complete: bool
    issues: list[FileIssue] = field(default_factory=list)
    modalities: dict[str, dict[str, Any]] = field(default_factory=dict)
    label_values: list[int] = field(default_factory=list)
    shapes_match: bool | None = None
    spacing_match: bool | None = None
    orientation_match: bool | None = None

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["issues"] = [asdict(issue) for issue in self.issues]
        return payload


@dataclass
class ValidationReport:
    n_patients: int
    n_ok: int
    n_failed: int
    n_missing_files: int
    n_corrupt: int
    patients: list[PatientValidation]
    notes: list[str] = field(default_factory=list)

    def ok_ids(self) -> list[str]:
        return [row.patient_id for row in self.patients if row.ok]

    def to_dict(self) -> dict[str, Any]:
        return {
            "n_patients": self.n_patients,
            "n_ok": self.n_ok,
            "n_failed": self.n_failed,
            "n_missing_files": self.n_missing_files,
            "n_corrupt": self.n_corrupt,
            "notes": self.notes,
            "patients": [row.to_dict() for row in self.patients],
            "clinical_disclaimer": (
                "Research prototype. Not a clinically validated diagnostic tool."
            ),
        }


def _issue(key: str, path: str, code: str, message: str) -> FileIssue:
    return FileIssue(key=key, path=path, code=code, message=message)


def _read_info(key: str, path: Path, issues: list[FileIssue]) -> NiftiInfo | None:
    try:
        return nifti_info(path)
    except CorruptedNiftiError as exc:
        issues.append(_issue(key, str(path), "corrupt_header", str(exc)))
        return None
    except Exception as exc:  # unexpected decoder failures still count as corrupt
        issues.append(_issue(key, str(path), "corrupt_header", str(exc)))
        return None


def _read_array(key: str, path: Path, issues: list[FileIssue]) -> np.ndarray | None:
    try:
        return load_nifti(path)
    except CorruptedNiftiError as exc:
        issues.append(_issue(key, str(path), "corrupt_data", str(exc)))
        return None
    except Exception as exc:
        issues.append(_issue(key, str(path), "corrupt_data", str(exc)))
        return None


def validate_case(
    case: PatientCase,
    *,
    expect_ndim: int = 3,
    spacing_atol_mm: float = 0.15,
    target_spacing_mm: tuple[float, float, float] | None = (1.0, 1.0, 1.0),
    require_complete: bool = True,
    load_voxels: bool = True,
) -> PatientValidation:
    issues: list[FileIssue] = []
    modalities: dict[str, dict[str, Any]] = {}
    infos: dict[str, NiftiInfo] = {}
    arrays: dict[str, np.ndarray] = {}

    for key, missing_path in ((item, "") for item in case.missing):
        issues.append(_issue(key, missing_path, "missing_file", f"No {key} file for {case.patient_id}"))

    for key, path_str in case.files.items():
        path = Path(path_str)
        if not path.is_file():
            issues.append(_issue(key, path_str, "missing_file", "Path does not exist"))
            continue
        if path.stat().st_size == 0:
            issues.append(_issue(key, path_str, "corrupt_empty", "File size is 0 bytes"))
            continue
        info = _read_info(key, path, issues)
        if info is None:
            continue
        infos[key] = info
        modalities[key] = asdict(info)
        if info.ndim != expect_ndim:
            issues.append(
                _issue(key, path_str, "ndim", f"Expected {expect_ndim}D, got shape {info.shape}")
            )
        if load_voxels:
            data = _read_array(key, path, issues)
            if data is None:
                continue
            arrays[key] = data
            if data.shape != info.shape:
                issues.append(
                    _issue(key, path_str, "shape_header", "Header shape does not match array")
                )
            if not np.isfinite(data).all():
                issues.append(_issue(key, path_str, "nonfinite", "NaN or Inf voxels"))
            if key in IMAGE_MODALITIES and not np.any(data != 0):
                issues.append(_issue(key, path_str, "all_zero", "MRI volume is all zeros"))

    image_infos = [infos[key] for key in IMAGE_MODALITIES if key in infos]
    shapes_match = None
    spacing_match = None
    orientation_match = None
    if len(image_infos) >= 2:
        shapes_match = len({info.shape for info in image_infos}) == 1
        if not shapes_match:
            issues.append(
                _issue("image", case.folder, "shape_mismatch", "MRI modalities do not share shape")
            )
        zooms = [info.zooms_mm for info in image_infos]
        spacing_match = all(
            np.allclose(zoom, zooms[0], atol=spacing_atol_mm) for zoom in zooms[1:]
        )
        if not spacing_match:
            issues.append(
                _issue("image", case.folder, "spacing_mismatch", f"Inconsistent zooms: {zooms}")
            )
        orientation_match = len({info.axcodes for info in image_infos}) == 1
        if not orientation_match:
            issues.append(
                _issue(
                    "image",
                    case.folder,
                    "orientation_mismatch",
                    f"Inconsistent axcodes: {[info.axcodes for info in image_infos]}",
                )
            )
    if target_spacing_mm and image_infos:
        for info in image_infos:
            if len(info.zooms_mm) >= 3 and not np.allclose(
                info.zooms_mm[:3], target_spacing_mm, atol=spacing_atol_mm
            ):
                issues.append(
                    _issue(
                        Path(info.path).name,
                        info.path,
                        "spacing_off_target",
                        f"zooms={info.zooms_mm} target={target_spacing_mm}",
                    )
                )

    if "seg" in infos and "seg" in arrays:
        if image_infos and arrays["seg"].shape != image_infos[0].shape:
            issues.append(
                _issue("seg", case.files["seg"], "seg_shape", "Segmentation shape differs from MRI")
            )
        labels = remap_brats_labels(arrays["seg"])
        unique = sorted({int(v) for v in np.unique(labels)})
        unexpected = [value for value in unique if value not in ALLOWED_LABELS]
        if unexpected:
            issues.append(
                _issue("seg", case.files["seg"], "unexpected_labels", f"Values {unexpected}")
            )
        label_values = unique
    else:
        label_values = []

    complete = not case.missing
    blocking = [
        issue
        for issue in issues
        if issue.code
        not in {"spacing_off_target"}  # off-target spacing is a warning; resample later
    ]
    if require_complete:
        ok = complete and not blocking
    else:
        ok = not blocking and bool(infos)
    if not load_voxels:
        ok = complete and not any(issue.code.startswith("corrupt") or issue.code == "missing_file" for issue in issues)

    return PatientValidation(
        patient_id=case.patient_id,
        ok=ok,
        complete=complete,
        issues=issues,
        modalities=modalities,
        label_values=label_values,
        shapes_match=shapes_match,
        spacing_match=spacing_match,
        orientation_match=orientation_match,
    )


def validate_cases(
    cases: list[PatientCase],
    **kwargs: Any,
) -> ValidationReport:
    rows = [validate_case(case, **kwargs) for case in cases]
    n_missing = sum(1 for row in rows if any(issue.code == "missing_file" for issue in row.issues))
    n_corrupt = sum(
        1
        for row in rows
        if any(issue.code.startswith("corrupt") for issue in row.issues)
    )
    n_ok = sum(1 for row in rows if row.ok)
    notes = []
    if n_ok == 0:
        notes.append("No patient passed validation. Training must not start.")
    return ValidationReport(
        n_patients=len(rows),
        n_ok=n_ok,
        n_failed=len(rows) - n_ok,
        n_missing_files=n_missing,
        n_corrupt=n_corrupt,
        patients=rows,
        notes=notes,
    )


def require_valid_case(case: PatientCase, **kwargs: Any) -> PatientValidation:
    row = validate_case(case, **kwargs)
    if not row.ok:
        messages = "; ".join(f"{issue.key}:{issue.code}" for issue in row.issues) or "unknown"
        raise DatasetValidationError(f"{case.patient_id} failed validation ({messages})")
    return row
