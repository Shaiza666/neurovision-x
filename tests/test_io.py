"""Filename matching for BraTS-style folders. Empty files are not volumes."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from neurovisionx.data.dataset import MissingBraTSDataError, require_brats_root
from neurovisionx.data.io import find_modality, load_nifti, resolve_patient_files, save_nifti


def _touch(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"")


class IoTests(unittest.TestCase):
    def test_brats2021_suffixes(self):
        with tempfile.TemporaryDirectory() as tmp:
            patient = Path(tmp) / "BraTS2021_00000"
            for name in ("t1", "t1ce", "t2", "flair", "seg"):
                _touch(patient / f"BraTS2021_00000_{name}.nii.gz")
            files = resolve_patient_files(patient)
            self.assertTrue(files["t1"].name.endswith("_t1.nii.gz"))
            self.assertTrue(files["t1ce"].name.endswith("_t1ce.nii.gz"))

    def test_brats2023_suffixes(self):
        with tempfile.TemporaryDirectory() as tmp:
            patient = Path(tmp) / "BraTS-GLI-00000-000"
            for token in ("t1n", "t1c", "t2w", "t2f", "seg"):
                _touch(patient / f"BraTS-GLI-00000-000-{token}.nii.gz")
            files = resolve_patient_files(patient)
            self.assertTrue(files["t1"].name.endswith("-t1n.nii.gz"))
            self.assertTrue(files["t1ce"].name.endswith("-t1c.nii.gz"))
            self.assertTrue(files["flair"].name.endswith("-t2f.nii.gz"))

    def test_t1_does_not_steal_t1ce(self):
        with tempfile.TemporaryDirectory() as tmp:
            patient = Path(tmp) / "case"
            _touch(patient / "case_t1.nii.gz")
            _touch(patient / "case_t1ce.nii.gz")
            t1 = find_modality(patient, "t1")
            t1ce = find_modality(patient, "t1ce")
            self.assertTrue(t1.name.endswith("_t1.nii.gz"))
            self.assertTrue(t1ce.name.endswith("_t1ce.nii.gz"))
            self.assertNotEqual(t1, t1ce)

    def test_empty_patient_dir_is_not_data(self):
        with tempfile.TemporaryDirectory() as tmp:
            raw = Path(tmp) / "brats"
            (raw / "patient").mkdir(parents=True)
            with self.assertRaises(MissingBraTSDataError):
                require_brats_root(raw)

    def test_save_and_load_nifti_roundtrip(self):
        try:
            import nibabel  # noqa: F401
            import numpy as np
        except ImportError:
            self.skipTest("nibabel/numpy missing")
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "pred.nii.gz"
            data = np.zeros((4, 4, 4), dtype=np.int16)
            data[1:3, 1:3, 1:3] = 3
            save_nifti(path, data)
            loaded = load_nifti(path)
            np.testing.assert_array_equal(loaded, data)

    def test_loose_files_without_patient_dirs_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            raw = Path(tmp) / "brats"
            raw.mkdir()
            _touch(raw / "volume.nii.gz")
            with self.assertRaises(MissingBraTSDataError):
                require_brats_root(raw)


if __name__ == "__main__":
    unittest.main()
