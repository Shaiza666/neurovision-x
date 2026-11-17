"""Attention U-Net shape tests. Random tensors are not medical volumes."""

from __future__ import annotations

import unittest


class AttentionUNetTests(unittest.TestCase):
    def test_forward_shape_matches_unet(self):
        try:
            import torch
        except ImportError:
            self.skipTest("torch not installed")
        from neurovisionx.models.attention_unet3d import AttentionUNet3D
        from neurovisionx.models.unet3d import UNet3D

        kwargs = dict(in_channels=4, out_channels=4, features=(8, 16), dropout=0.0)
        volume = torch.randn(1, 4, 16, 16, 16)
        unet = UNet3D(**kwargs)
        attn = AttentionUNet3D(**kwargs)
        self.assertEqual(tuple(unet(volume).shape), (1, 4, 16, 16, 16))
        self.assertEqual(tuple(attn(volume).shape), (1, 4, 16, 16, 16))

    def test_factory_builds_attention(self):
        try:
            import torch  # noqa: F401
        except ImportError:
            self.skipTest("torch not installed")
        from neurovisionx.models.attention_unet3d import AttentionUNet3D
        from neurovisionx.models.factory import build_model

        model = build_model(
            {
                "name": "attention_unet3d",
                "in_channels": 4,
                "out_channels": 4,
                "features": [8, 16],
                "optional_monai": False,
            }
        )
        self.assertIsInstance(model, AttentionUNet3D)
