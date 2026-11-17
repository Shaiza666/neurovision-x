"""Stats and QC tests on fixture NIfTI (not a clinical report)."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from neurovisionx.data.inventory import discover_inventory
from neurovisionx.data.preprocess import preprocess_patient_files, stack_modalities
from neurovisionx.data.qc import save_class_histogram, save_patient_qc
from neurovisionx.data.stats import aggregate_stats, patient_stats
from tests.nifti_fixtures import write_mini_dataset, write_patient


class StatsQcTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            import matplotlib  # noqa: F401
            import nibabel  # noqa: F401
        except ImportError:
            raise unittest.SkipTest("nibabel/matplotlib not installed")

    def test_class_counts_include_tumor(self):
        with tempfile.TemporaryDirectory() as tmp:
            write_patient(Path(tmp) / "p0", "BraTS2021_00000")
            case = discover_inventory(Path(tmp)).cases[0]
            stats = patient_stats(case)
            self.assertGreater(stats.class_voxels["ncr"], 0)
            self.assertGreater(stats.region_voxels["WT"], 0)
            payload = aggregate_stats([stats])
            self.assertEqual(payload["n_patients"], 1)
            self.assertGreater(payload["class_fractions"]["background"], 0.5)

    def test_qc_writes_png(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = write_patient(Path(tmp) / "p0", "BraTS2021_00000")
            from neurovisionx.data.io import find_available_modalities

            arrays = preprocess_patient_files(find_available_modalities(folder), crop_foreground=False)
            image = stack_modalities(arrays)
            dest = Path(tmp) / "qc.png"
            save_patient_qc(image, arrays["seg"], dest, title="fixture")
            self.assertTrue(dest.is_file())
            self.assertGreater(dest.stat().st_size, 100)
            hist = save_class_histogram({"background": 10, "et": 2}, Path(tmp) / "hist.png")
            self.assertTrue(hist.is_file())

    def test_prepare_splits_script_on_fixture(self):
        import subprocess
        import sys

        from neurovisionx.utils.paths import project_root

        with tempfile.TemporaryDirectory() as tmp:
            root = write_mini_dataset(Path(tmp) / "brats", n_patients=6)
            out = Path(tmp) / "patients.json"
            script = project_root() / "scripts" / "prepare_splits.py"
            completed = subprocess.run(
                [sys.executable, str(script), "--raw-root", str(root), "--out", str(out)],
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)
            self.assertTrue(out.is_file())
            from neurovisionx.data.splits import assert_no_leakage, load_split

            split = load_split(out)
            assert_no_leakage(split)
            self.assertEqual(len(split.all_ids()), 6)
