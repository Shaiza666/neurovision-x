"""Loss and metric tests on synthetic tensors (not medical data)."""

from __future__ import annotations

import unittest

import numpy as np

from neurovisionx.evaluation.metrics import dice_coefficient, expected_calibration_error


class MetricTests(unittest.TestCase):
    def test_perfect_dice(self):
        mask = np.zeros((8, 8, 8), dtype=np.uint8)
        mask[2:5, 2:5, 2:5] = 1
        self.assertEqual(dice_coefficient(mask, mask), 1.0)

    def test_empty_agreement(self):
        empty = np.zeros((4, 4, 4), dtype=np.uint8)
        self.assertEqual(dice_coefficient(empty, empty), 1.0)

    def test_hd95_nan_when_empty(self):
        try:
            from neurovisionx.evaluation.metrics import hd95
        except ImportError:
            self.skipTest("hd95 import failed")
        empty = np.zeros((6, 6, 6), dtype=np.uint8)
        blob = np.zeros_like(empty)
        blob[2:4, 2:4, 2:4] = 1
        try:
            self.assertTrue(np.isnan(hd95(empty, blob)))
        except ImportError:
            self.skipTest("scipy not installed")

    def test_ece_perfectly_calibrated(self):
        confidence = np.array([0.0, 0.0, 1.0, 1.0])
        correct = np.array([0.0, 0.0, 1.0, 1.0])
        self.assertLess(expected_calibration_error(confidence, correct, n_bins=2), 0.05)


class TorchLossTests(unittest.TestCase):
    def test_dice_zero_on_one_hot_identity(self):
        try:
            import torch
        except ImportError:
            self.skipTest("torch not installed")
        from neurovisionx.losses.segmentation import CombinedSegLoss, dice_loss

        target = torch.zeros(1, 8, 8, 8, dtype=torch.long)
        target[:, 2:6, 2:6, 2:6] = 1
        logits = torch.zeros(1, 4, 8, 8, 8)
        logits[0, 1, 2:6, 2:6, 2:6] = 10
        logits[0, 0] = 10
        logits[0, 0, 2:6, 2:6, 2:6] = 0
        loss = dice_loss(logits, target, include_background=False)
        self.assertLess(float(loss), 0.05)
        combined = CombinedSegLoss(dice_weight=1.0, ce_weight=0.0, boundary_weight=0.0)
        self.assertGreater(float(combined(logits, target)), -1e-6)


if __name__ == "__main__":
    unittest.main()
