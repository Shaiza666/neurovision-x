"""Research demo backend tests. Uses the real smoke checkpoint when present."""

from __future__ import annotations

import json
import tempfile
import unittest
import zipfile
from io import BytesIO
from pathlib import Path

import numpy as np

from neurovisionx.demo.uploads import UploadValidationError, validate_modality_paths
from neurovisionx.utils.paths import project_root


class UploadValidationTests(unittest.TestCase):
    def test_missing_modality_is_explicit(self):
        with self.assertRaises(UploadValidationError) as ctx:
            validate_modality_paths({"t1": Path("t1.nii.gz"), "t1ce": None, "t2": None, "flair": None})
        self.assertIn("Missing required modalities", str(ctx.exception))
        self.assertIn("t1ce", str(ctx.exception))
        self.assertIn("Empty channels are not synthesized", str(ctx.exception))

    def test_non_nifti_rejected(self):
        from neurovisionx.demo.uploads import write_named_bytes

        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(UploadValidationError):
                write_named_bytes(Path(tmp), "photo.jpg", b"not-nifti")

    def test_shape_mismatch_rejected(self):
        try:
            import nibabel  # noqa: F401
        except ImportError:
            self.skipTest("nibabel not installed")
        from neurovisionx.demo.uploads import load_validated_case
        from tests.nifti_fixtures import write_nifti, brain_volume

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_nifti(root / "t1.nii.gz", brain_volume((12, 12, 12)))
            write_nifti(root / "t1ce.nii.gz", brain_volume((12, 12, 12), 1.0))
            write_nifti(root / "t2.nii.gz", brain_volume((12, 12, 12), 2.0))
            write_nifti(root / "flair.nii.gz", brain_volume((8, 8, 8), 3.0))
            with self.assertRaises(UploadValidationError) as ctx:
                load_validated_case(
                    {
                        "t1": root / "t1.nii.gz",
                        "t1ce": root / "t1ce.nii.gz",
                        "t2": root / "t2.nii.gz",
                        "flair": root / "flair.nii.gz",
                    }
                )
            self.assertIn("not co-registered", str(ctx.exception))

    def test_valid_four_modalities_and_seg(self):
        try:
            import nibabel  # noqa: F401
        except ImportError:
            self.skipTest("nibabel not installed")
        from neurovisionx.demo.uploads import load_validated_case
        from tests.nifti_fixtures import write_patient

        with tempfile.TemporaryDirectory() as tmp:
            folder = write_patient(Path(tmp) / "case", "demo", shape=(16, 16, 16))
            files = {
                "t1": folder / "demo_t1.nii.gz",
                "t1ce": folder / "demo_t1ce.nii.gz",
                "t2": folder / "demo_t2.nii.gz",
                "flair": folder / "demo_flair.nii.gz",
                "seg": folder / "demo_seg.nii.gz",
            }
            case = load_validated_case(files)
            self.assertEqual(case["image"].shape, (4, 16, 16, 16))
            self.assertEqual(case["seg"].shape, (16, 16, 16))
            self.assertNotIn(4, np.unique(case["seg"]))


class CheckpointConfigTests(unittest.TestCase):
    def test_smoke_payload_uses_tiny_features_not_yaml_widths(self):
        from neurovisionx.demo.checkpoints import model_cfg_from_payload, patch_size_from_payload

        payload = {
            "smoke": True,
            "config": {
                "model": {
                    "name": "unet3d",
                    "features": [16, 32, 64, 128],
                    "optional_monai": True,
                    "in_channels": 4,
                    "out_channels": 4,
                },
                "training": {"smoke": {"features": [8, 16], "patch_size": [16, 16, 16]}},
            },
        }
        cfg = model_cfg_from_payload(payload, fallback=payload["config"]["model"])
        self.assertEqual(cfg["features"], [8, 16])
        self.assertFalse(cfg["optional_monai"])
        self.assertEqual(patch_size_from_payload(payload), (16, 16, 16))

    def test_missing_checkpoint_is_an_error(self):
        try:
            import torch  # noqa: F401
        except ImportError:
            self.skipTest("torch not installed")
        from neurovisionx.demo.checkpoints import load_demo_model

        with self.assertRaises(FileNotFoundError):
            load_demo_model(Path("experiments/does_not_exist/best.pt"))


class SmokeCheckpointInferenceTests(unittest.TestCase):
    def test_list_includes_on_disk_smoke_weights(self):
        ckpt = project_root() / "experiments" / "smoke" / "best.pt"
        if not ckpt.is_file():
            self.skipTest("smoke checkpoint not on disk")
        from neurovisionx.demo.checkpoints import list_demo_checkpoints

        rows = list_demo_checkpoints()
        paths = [row["path"] for row in rows]
        self.assertIn(ckpt, paths)
        smoke_row = next(row for row in rows if row["path"] == ckpt)
        self.assertTrue(smoke_row["smoke"])
        self.assertIn("SMOKE", smoke_row["label"])

    def test_smoke_weights_produce_model_logits_not_mocks(self):
        try:
            import torch
        except ImportError:
            self.skipTest("torch not installed")
        ckpt = project_root() / "experiments" / "smoke" / "best.pt"
        if not ckpt.is_file():
            self.skipTest("smoke checkpoint not on disk")
        from neurovisionx.demo.checkpoints import load_demo_model, patch_size_from_payload
        from neurovisionx.demo.pipeline import bundle_zip, run_inference

        model, payload = load_demo_model(ckpt, map_location="cpu")
        self.assertTrue(payload.get("smoke"))
        patch = patch_size_from_payload(payload)
        torch.manual_seed(0)
        image = torch.randn(4, *patch).numpy()
        target = np.zeros(patch, dtype=np.int64)
        d, h, w = patch
        target[d // 4 : d // 2, h // 4 : h // 2, w // 4 : w // 2] = 1
        result = run_inference(model, payload, image, target=target, compute_cam=True, class_index=3)
        self.assertTrue(result["smoke"])
        self.assertIn("SMOKE", result["status_banner"])
        self.assertEqual(tuple(result["prediction"].shape), tuple(patch))
        volume = torch.from_numpy(np.ascontiguousarray(image)).unsqueeze(0)
        model.eval()
        with torch.no_grad():
            expected = model(volume).argmax(dim=1)[0].cpu().numpy()
        np.testing.assert_array_equal(result["prediction"], expected)
        self.assertGreater(result["elapsed_ms"], 0.0)
        self.assertGreater(result["n_params"], 0)
        self.assertIsNotNone(result["cam"])
        self.assertEqual(tuple(result["entropy"].shape), tuple(patch))
        self.assertNotIn("mock", result["status_banner"].lower())
        self.assertIsNotNone(result["scores"])
        self.assertIn("dice_wt", result["scores"])

        with tempfile.TemporaryDirectory() as tmp:
            zipped = bundle_zip(image, result, target=target, out_dir=Path(tmp) / "bundle")
        with zipfile.ZipFile(BytesIO(zipped)) as zf:
            names = set(zf.namelist())
        self.assertIn("inference.json", names)
        self.assertIn("orthogonal_overlay.png", names)
        sidecar = json.loads(zipfile.ZipFile(BytesIO(zipped)).read("inference.json"))
        self.assertTrue(sidecar["smoke"])
        self.assertEqual(sidecar["n_params"], result["n_params"])


if __name__ == "__main__":
    unittest.main()
