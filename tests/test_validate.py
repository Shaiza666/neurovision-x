"""Validation tests on fixture NIfTI (not reported as BraTS metrics)."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from neurovisionx.data.inventory import discover_inventory
from neurovisionx.data.validate import validate_case, validate_cases
from tests.nifti_fixtures import write_nifti, write_patient


class ValidateTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            import nibabel  # noqa: F401
        except ImportError:
            raise unittest.SkipTest("nibabel not installed")

    def test_complete_patient_ok(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = write_patient(Path(tmp) / "p0", "BraTS2021_00000")
            case = discover_inventory(folder.parent).cases[0]
            row = validate_case(case)
            self.assertTrue(row.ok)
            self.assertTrue(row.complete)
            self.assertEqual(row.shapes_match, True)
            self.assertIn(3, row.label_values)
            self.assertNotIn(4, row.label_values)

    def test_missing_modality(self):
        with tempfile.TemporaryDirectory() as tmp:
            write_patient(Path(tmp) / "p0", "BraTS2021_00000", skip=("t2",))
            case = discover_inventory(Path(tmp)).cases[0]
            row = validate_case(case)
            self.assertFalse(row.ok)
            self.assertIn("t2", case.missing)
            self.assertTrue(any(issue.code == "missing_file" for issue in row.issues))

    def test_corrupt_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            write_patient(Path(tmp) / "p0", "BraTS2021_00000", corrupt=("flair",))
            case = discover_inventory(Path(tmp)).cases[0]
            row = validate_case(case)
            self.assertFalse(row.ok)
            self.assertTrue(any("corrupt" in issue.code for issue in row.issues))

    def test_shape_mismatch(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = write_patient(Path(tmp) / "p0", "BraTS2021_00000")
            write_nifti(folder / "BraTS2021_00000_t1.nii.gz", np.zeros((10, 10, 10), dtype=np.float32))
            case = discover_inventory(Path(tmp)).cases[0]
            row = validate_case(case)
            self.assertFalse(row.ok)
            self.assertTrue(any(issue.code in {"shape_mismatch", "seg_shape"} for issue in row.issues))

    def test_report_counts(self):
        with tempfile.TemporaryDirectory() as tmp:
            write_patient(Path(tmp) / "ok", "BraTS2021_00001")
            write_patient(Path(tmp) / "bad", "BraTS2021_00002", skip=("seg",))
            report = validate_cases(discover_inventory(Path(tmp)).cases)
            self.assertEqual(report.n_patients, 2)
            self.assertEqual(report.n_ok, 1)


if __name__ == "__main__":
    unittest.main()
