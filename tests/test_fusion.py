"""SE fusion wrapper tests. Random tensors are not medical volumes."""

from __future__ import annotations

import unittest


class FusionTests(unittest.TestCase):
    def test_early_fusion_is_plain_unet(self):
        try:
            import torch  # noqa: F401
        except ImportError:
            self.skipTest("torch not installed")
        from neurovisionx.models.factory import build_model
        from neurovisionx.models.unet3d import UNet3D

        model = build_model(
            {
                "name": "unet3d",
                "in_channels": 4,
                "out_channels": 4,
                "features": [8, 16],
                "optional_monai": False,
                "fusion": "early",
            }
        )
        self.assertIsInstance(model, UNet3D)

    def test_se_fusion_wraps_without_editing_unet(self):
        try:
            import torch
        except ImportError:
            self.skipTest("torch not installed")
        from neurovisionx.models.factory import build_model
        from neurovisionx.models.fusion import FusionStem
        from neurovisionx.models.unet3d import UNet3D

        model = build_model(
            {
                "name": "unet3d",
                "in_channels": 4,
                "out_channels": 4,
                "features": [8, 16],
                "optional_monai": False,
                "fusion": "se",
            }
        )
        self.assertIsInstance(model, FusionStem)
        self.assertIsInstance(model.backbone, UNet3D)
        volume = torch.randn(1, 4, 16, 16, 16)
        logits = model(volume)
        self.assertEqual(tuple(logits.shape), (1, 4, 16, 16, 16))
        self.assertGreater(
            sum(p.numel() for p in model.parameters()),
            sum(p.numel() for p in model.backbone.parameters()),
        )

    def test_unet3d_source_has_no_fusion_stem(self):
        from neurovisionx.utils.paths import project_root

        source = (project_root() / "src" / "neurovisionx" / "models" / "unet3d.py").read_text(
            encoding="utf-8"
        )
        self.assertNotIn("FusionStem", source)
        self.assertNotIn("SqueezeExcite", source)

    def test_unknown_fusion_raises(self):
        try:
            import torch  # noqa: F401
        except ImportError:
            self.skipTest("torch not installed")
        from neurovisionx.models.factory import build_model

        with self.assertRaises(ValueError):
            build_model(
                {
                    "name": "unet3d",
                    "features": [8, 16],
                    "optional_monai": False,
                    "fusion": "cross_attention_novel",
                }
            )


if __name__ == "__main__":
    unittest.main()
