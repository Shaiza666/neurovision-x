"""Explainability tests on synthetic tensors. Not MRI. Not research figures."""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from neurovisionx.utils.paths import project_root


class GradCAMTests(unittest.TestCase):
    def test_cam_matches_prediction_shape_and_is_nonnegative(self):
        try:
            import torch
        except ImportError:
            self.skipTest("torch not installed")
        from neurovisionx.explainability.gradcam import gradcam_3d
        from neurovisionx.models.factory import build_model

        model = build_model(
            {
                "name": "unet3d",
                "in_channels": 4,
                "out_channels": 4,
                "features": [8, 16],
                "dropout": 0.0,
                "optional_monai": False,
                "fusion": "early",
            }
        )
        volume = torch.randn(1, 4, 16, 16, 16)
        result = gradcam_3d(model, volume, class_index=3)
        self.assertEqual(tuple(result.prediction.shape), (16, 16, 16))
        self.assertEqual(tuple(result.cam.shape), (16, 16, 16))
        self.assertEqual(tuple(result.logits.shape), (4, 16, 16, 16))
        self.assertTrue(torch.all(result.cam >= 0))
        self.assertTrue(torch.isfinite(result.cam).all())
        model.eval()
        with torch.no_grad():
            again = model(volume).argmax(dim=1)[0]
        self.assertTrue(torch.equal(again, result.prediction))
        self.assertIn(result.score_mode, {"predicted_region_mean", "class_logit_mean_empty_pred"})
        self.assertIn("not a clinical explanation", result.disclaimer.lower())

    def test_se_wrapper_still_has_conv3d_target(self):
        try:
            import torch
        except ImportError:
            self.skipTest("torch not installed")
        from neurovisionx.explainability.gradcam import select_cam_layer
        from neurovisionx.models.factory import build_model
        from neurovisionx.models.fusion import FusionStem

        model = build_model(
            {
                "name": "unet3d",
                "features": [8, 16],
                "optional_monai": False,
                "fusion": "se",
            }
        )
        self.assertIsInstance(model, FusionStem)
        name, layer = select_cam_layer(model)
        self.assertGreater(int(layer.kernel_size[0]), 1)
        self.assertTrue(name)


class OcclusionTests(unittest.TestCase):
    def test_occlusion_map_shape_nonnegative(self):
        try:
            import torch
        except ImportError:
            self.skipTest("torch not installed")
        from neurovisionx.explainability.occlusion import occlusion_sensitivity
        from neurovisionx.models.factory import build_model

        model = build_model(
            {
                "name": "unet3d",
                "in_channels": 4,
                "out_channels": 4,
                "features": [8, 16],
                "dropout": 0.0,
                "optional_monai": False,
                "fusion": "early",
            }
        )
        volume = torch.randn(1, 4, 16, 16, 16)
        sensitivity = occlusion_sensitivity(
            model, volume, patch=8, stride=8, class_index=3
        )
        self.assertEqual(tuple(sensitivity.shape), (16, 16, 16))
        self.assertTrue(torch.all(sensitivity >= 0))
        self.assertTrue(torch.isfinite(sensitivity).all())

    def test_explain_volume_writes_occlusion_when_requested(self):
        try:
            import torch
        except ImportError:
            self.skipTest("torch not installed")
        from neurovisionx.explainability.panel import explain_volume
        from neurovisionx.models.factory import build_model

        model = build_model(
            {
                "name": "unet3d",
                "features": [8, 16],
                "dropout": 0.0,
                "optional_monai": False,
                "fusion": "early",
            }
        )
        volume = torch.randn(1, 4, 16, 16, 16)
        with tempfile.TemporaryDirectory() as tmp:
            paths = explain_volume(
                model,
                volume,
                volume[0].numpy(),
                tmp,
                class_index=3,
                also_occlusion=True,
                title="SYNTHETIC RANDOM TENSORS — not MRI.",
            )
            sidecar = json.loads(Path(paths["sidecar"]).read_text(encoding="utf-8"))
            self.assertTrue(sidecar["has_occlusion"])
            self.assertTrue(Path(paths["occlusion"]).is_file())


class OverlayAndViewTests(unittest.TestCase):
    def test_three_orientations_and_error_legend_png(self):
        try:
            import matplotlib  # noqa: F401
            import numpy as np
        except ImportError:
            self.skipTest("numpy/matplotlib missing")
        from neurovisionx.explainability.overlays import save_orthogonal_overlay
        from neurovisionx.explainability.views import ORIENTATION_ORDER, extract_slice
        from neurovisionx.explainability.volume3d import save_prediction_pointcloud

        image = np.random.randn(4, 16, 16, 16)
        pred = np.zeros((16, 16, 16), dtype=np.int16)
        pred[4:10, 4:10, 4:10] = 3
        pred[6:12, 6:12, 6:12] = 1
        target = np.zeros_like(pred)
        target[5:11, 5:11, 5:11] = 3
        for name in ORIENTATION_ORDER:
            plane = extract_slice(pred, name)
            self.assertEqual(plane.ndim, 2)
        with tempfile.TemporaryDirectory() as tmp:
            overlay = Path(tmp) / "overlay.png"
            cloud = Path(tmp) / "cloud.png"
            save_orthogonal_overlay(image, pred, overlay, target=target)
            save_prediction_pointcloud(pred, cloud)
            self.assertGreater(overlay.stat().st_size, 0)
            self.assertGreater(cloud.stat().st_size, 0)

    def test_panel_writes_sidecar_from_same_pass(self):
        try:
            import torch
        except ImportError:
            self.skipTest("torch not installed")
        import numpy as np

        from neurovisionx.explainability.panel import explain_volume
        from neurovisionx.models.factory import build_model

        model = build_model(
            {
                "name": "unet3d",
                "features": [8, 16],
                "dropout": 0.0,
                "optional_monai": False,
                "fusion": "early",
            }
        )
        volume = torch.randn(1, 4, 16, 16, 16)
        target = np.zeros((16, 16, 16), dtype=np.int16)
        target[4:8, 4:8, 4:8] = 3
        with tempfile.TemporaryDirectory() as tmp:
            paths = explain_volume(
                model,
                volume,
                volume[0].numpy(),
                tmp,
                target=target,
                class_index=3,
                title="SYNTHETIC RANDOM TENSORS — not MRI. Research visualization.",
            )
            sidecar = json.loads(Path(paths["sidecar"]).read_text(encoding="utf-8"))
            self.assertTrue(sidecar["has_ground_truth"])
            self.assertEqual(sidecar["entropy_kind"], "softmax_entropy_T1_same_forward")
            self.assertIn("not a clinical explanation", sidecar["disclaimer"].lower())
            for key in ("overlay", "attribution", "volume3d"):
                self.assertTrue(Path(paths[key]).is_file())


class ExplainScriptGateTests(unittest.TestCase):
    def test_explain_exits_without_data(self):
        script = project_root() / "scripts" / "explain.py"
        completed = subprocess.run(
            [sys.executable, str(script), "--patient-id", "BraTSX"],
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertNotEqual(completed.returncode, 0)
        combined = completed.stdout + completed.stderr
        self.assertIn("No BraTS volumes", combined)

    def test_synthetic_flag_writes_watermarked_panel(self):
        script = project_root() / "scripts" / "explain.py"
        with tempfile.TemporaryDirectory() as tmp:
            completed = subprocess.run(
                [sys.executable, str(script), "--synthetic", "--out-dir", tmp],
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)
            sidecar = json.loads((Path(tmp) / "explain.json").read_text(encoding="utf-8"))
            self.assertIn("not a clinical explanation", sidecar["disclaimer"].lower())
            self.assertTrue((Path(tmp) / "orthogonal_overlay.png").is_file())


if __name__ == "__main__":
    unittest.main()
