"""3D U-Net shape tests. Random tensors are not medical volumes."""

from __future__ import annotations

import unittest


class UNetShapeTests(unittest.TestCase):
    def test_forward_shape(self):
        try:
            import torch
        except ImportError:
            self.skipTest("torch not installed")
        from neurovisionx.models.unet3d import UNet3D

        model = UNet3D(in_channels=4, out_channels=4, features=(8, 16), dropout=0.1)
        volume = torch.randn(1, 4, 16, 16, 16)
        logits = model(volume)
        self.assertEqual(tuple(logits.shape), (1, 4, 16, 16, 16))

    def test_unetr_is_not_silently_substituted(self):
        try:
            import torch  # noqa: F401
        except ImportError:
            self.skipTest("torch not installed")
        from neurovisionx.models.factory import build_model

        with self.assertRaises(NotImplementedError):
            build_model({"name": "unetr", "in_channels": 4, "out_channels": 4})

    def test_missing_brats_is_explicit(self):
        from pathlib import Path

        from neurovisionx.data.dataset import MissingBraTSDataError, require_brats_root

        with self.assertRaises(MissingBraTSDataError):
            require_brats_root(Path("data/raw/brats"))


if __name__ == "__main__":
    unittest.main()
