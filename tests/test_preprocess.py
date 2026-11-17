"""Preprocess tests on synthetic arrays (not medical volumes)."""

from __future__ import annotations

import unittest

import numpy as np

from neurovisionx.data.preprocess import crop_or_pad, zscore_nonzero


class PreprocessTests(unittest.TestCase):
    def test_zscore_nonzero_ignores_zeros(self):
        volume = np.zeros((4, 4, 4), dtype=np.float32)
        volume[1:3, 1:3, 1:3] = 5.0
        volume[1, 1, 1] = 7.0
        out = zscore_nonzero(volume)
        self.assertEqual(float(out[0, 0, 0]), 0.0)
        self.assertAlmostEqual(float(out[out != 0].mean()), 0.0, places=5)

    def test_all_zero_volume_refused(self):
        with self.assertRaises(ValueError):
            zscore_nonzero(np.zeros((3, 3, 3)))

    def test_center_crop(self):
        image = np.zeros((2, 8, 8, 8), dtype=np.float32)
        image[:, 3:5, 3:5, 3:5] = 1.0
        seg = np.zeros((8, 8, 8), dtype=np.int64)
        seg[3:5, 3:5, 3:5] = 1
        cropped_image, cropped_seg = crop_or_pad(image, seg, (4, 4, 4))
        self.assertEqual(cropped_image.shape, (2, 4, 4, 4))
        self.assertGreater(float(cropped_seg.sum()), 0)

    def test_pad_small_volume(self):
        image = np.ones((1, 2, 2, 2), dtype=np.float32)
        seg = np.ones((2, 2, 2), dtype=np.int64)
        cropped_image, cropped_seg = crop_or_pad(image, seg, (4, 4, 4))
        self.assertEqual(cropped_image.shape, (1, 4, 4, 4))
        self.assertEqual(int(cropped_seg.sum()), 8)


if __name__ == "__main__":
    unittest.main()
