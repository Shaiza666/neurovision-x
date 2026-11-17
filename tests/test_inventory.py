"""Discovery tests on temporary fixture trees (not BraTS research data)."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from neurovisionx.data.errors import MissingBraTSDataError
from neurovisionx.data.inventory import discover_inventory
from tests.nifti_fixtures import write_mini_dataset, write_patient


class InventoryTests(unittest.TestCase):
    def test_patient_folders_2021(self):
        try:
            import nibabel  # noqa: F401
        except ImportError:
            self.skipTest("nibabel not installed")
        with tempfile.TemporaryDirectory() as tmp:
            root = write_mini_dataset(Path(tmp) / "brats", n_patients=4, style="2021")
            inventory = discover_inventory(root)
            self.assertEqual(inventory.n_cases, 4)
            self.assertEqual(inventory.n_complete, 4)
            self.assertEqual(inventory.naming_counts.get("brats2021"), 4)
            self.assertEqual(inventory.layout, "patient_folders")

    def test_nested_official_subfolders(self):
        try:
            import nibabel  # noqa: F401
        except ImportError:
            self.skipTest("nibabel not installed")
        with tempfile.TemporaryDirectory() as tmp:
            root = write_mini_dataset(Path(tmp) / "brats", n_patients=4, style="2023", nested=True)
            inventory = discover_inventory(root)
            self.assertEqual(inventory.layout, "official_subfolders")
            self.assertIn("training", inventory.official_subdirs)
            self.assertIn("validation", inventory.official_subdirs)
            self.assertTrue(any(case.patient_id.startswith("training/") for case in inventory.cases))

    def test_flat_files_grouped_by_id(self):
        try:
            import nibabel  # noqa: F401
        except ImportError:
            self.skipTest("nibabel not installed")
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "flat"
            write_patient(root, "BraTS2021_00007", style="2021")
            write_patient(root, "BraTS2021_00008", style="2021")
            inventory = discover_inventory(root)
            self.assertEqual(inventory.n_complete, 2)
            self.assertIn("BraTS2021_00007", inventory.complete_ids())

    def test_missing_root(self):
        with self.assertRaises(MissingBraTSDataError):
            discover_inventory(Path("data/raw/brats-does-not-exist"))


if __name__ == "__main__":
    unittest.main()
