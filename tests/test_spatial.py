"""Spatial preprocess tests on fixture volumes (not medical research results)."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from tests.nifti_fixtures import brain_volume, write_nifti, write_patient


class SpatialTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            import nibabel  # noqa: F401
        except ImportError:
            raise unittest.SkipTest("nibabel not installed")

    def test_reorient_to_ras(self):
        import nibabel as nib

        from neurovisionx.data.spatial import canonical_ras

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "las.nii.gz"
            affine = np.diag([-1.0, 1.0, 1.0, 1.0])
            write_nifti(path, brain_volume(), affine)
            canonical = canonical_ras(path)
            codes = nib.aff2axcodes(canonical.affine)
            self.assertEqual(codes, ("R", "A", "S"))

    def test_aligned_stack_and_zscore(self):
        from neurovisionx.data.preprocess import preprocess_patient_files, stack_modalities

        with tempfile.TemporaryDirectory() as tmp:
            folder = write_patient(Path(tmp) / "p0", "BraTS2021_00000")
            from neurovisionx.data.io import find_available_modalities

            files = find_available_modalities(folder)
            arrays = preprocess_patient_files(files, crop_foreground=True)
            image = stack_modalities(arrays)
            self.assertEqual(image.shape[0], 4)
            self.assertEqual(image.shape[1:], arrays["seg"].shape)
            brain = image[0] != 0
            self.assertAlmostEqual(float(image[0][brain].mean()), 0.0, places=4)

    def test_foreground_crop_shrinks(self):
        from neurovisionx.data.spatial import load_aligned_patient
        from neurovisionx.data.io import find_available_modalities

        with tempfile.TemporaryDirectory() as tmp:
            folder = write_patient(Path(tmp) / "p0", "BraTS2021_00000", shape=(20, 20, 20))
            files = find_available_modalities(folder)
            cropped = load_aligned_patient(files, crop_foreground=True, margin=1)
            self.assertLess(cropped["t1"].shape[0], 20)
