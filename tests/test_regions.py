"""Nested BraTS region tests on synthetic class maps (not medical data)."""

from __future__ import annotations

import unittest

import numpy as np

from neurovisionx.evaluation.regions import nested_masks, region_scores, remap_brats_labels


class RegionTests(unittest.TestCase):
    def test_legacy_et_label_four_maps_to_three(self):
        label = np.array([0, 1, 2, 4])
        remapped = remap_brats_labels(label)
        np.testing.assert_array_equal(remapped, np.array([0, 1, 2, 3]))

    def test_nested_masks(self):
        volume = np.zeros((4, 4, 4), dtype=np.int64)
        volume[0, 0, 0] = 1  # NCR -> WT, TC
        volume[1, 1, 1] = 2  # ED -> WT
        volume[2, 2, 2] = 3  # ET -> WT, TC, ET
        masks = nested_masks(volume)
        self.assertEqual(int(masks["WT"].sum()), 3)
        self.assertEqual(int(masks["TC"].sum()), 2)
        self.assertEqual(int(masks["ET"].sum()), 1)

    def test_perfect_region_dice(self):
        volume = np.zeros((6, 6, 6), dtype=np.int64)
        volume[1:3, 1:3, 1:3] = 1
        volume[3:5, 3:5, 3:5] = 3
        scores = region_scores(volume, volume)
        self.assertEqual(scores["dice_wt"], 1.0)
        self.assertEqual(scores["dice_tc"], 1.0)
        self.assertEqual(scores["dice_et"], 1.0)

    def test_empty_vs_tumor_is_zero_dice(self):
        target = np.zeros((4, 4, 4), dtype=np.int64)
        target[1, 1, 1] = 3
        pred = np.zeros_like(target)
        scores = region_scores(pred, target)
        self.assertEqual(scores["dice_et"], 0.0)


if __name__ == "__main__":
    unittest.main()
