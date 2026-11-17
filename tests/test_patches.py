"""3D patch extraction tests on synthetic arrays (not MRI studies)."""

from __future__ import annotations

import unittest

import numpy as np

from neurovisionx.data.patches import extract_grid_patches, extract_training_patch


class PatchTests(unittest.TestCase):
    def test_grid_patch_shape(self):
        image = np.zeros((4, 20, 20, 20), dtype=np.float32)
        seg = np.zeros((20, 20, 20), dtype=np.int64)
        seg[8:12, 8:12, 8:12] = 1
        patches = extract_grid_patches(image, seg, (8, 8, 8), overlap=0.0)
        self.assertGreater(len(patches), 1)
        for patch_image, patch_seg, origin in patches:
            self.assertEqual(patch_image.shape, (4, 8, 8, 8))
            self.assertEqual(patch_seg.shape, (8, 8, 8))
            self.assertEqual(len(origin), 3)

    def test_foreground_patch_contains_tumor(self):
        image = np.zeros((2, 24, 24, 24), dtype=np.float32)
        seg = np.zeros((24, 24, 24), dtype=np.int64)
        seg[18:22, 18:22, 18:22] = 3
        rng = np.random.default_rng(0)
        hits = 0
        for _ in range(12):
            _, patch_seg = extract_training_patch(
                image, seg, (8, 8, 8), rng, foreground_prob=1.0
            )
            if int(patch_seg.max()) == 3:
                hits += 1
        self.assertGreater(hits, 8)

    def test_pad_small_volume(self):
        image = np.ones((1, 4, 4, 4), dtype=np.float32)
        seg = np.ones((4, 4, 4), dtype=np.int64)
        rng = np.random.default_rng(1)
        patch_image, patch_seg = extract_training_patch(image, seg, (8, 8, 8), rng)
        self.assertEqual(patch_image.shape, (1, 8, 8, 8))
        self.assertEqual(patch_seg.shape, (8, 8, 8))
