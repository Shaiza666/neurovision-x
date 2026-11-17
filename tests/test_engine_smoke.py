"""Synthetic smoke training. Numbers here are not BraTS research results."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path


class SmokeEngineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            import torch  # noqa: F401
        except ImportError:
            raise unittest.SkipTest("torch not installed")

    def test_one_epoch_writes_checkpoints_and_history(self):
        from neurovisionx.training.checkpoint import load_checkpoint, restore_training_state
        from neurovisionx.training.engine import fit
        from neurovisionx.training.loop import TrainRequest
        from neurovisionx.training.smoke import smoke_loaders, smoke_model_cfg
        from neurovisionx.utils.config import load_experiment_config
        from neurovisionx.utils.seed import seed_everything

        config = load_experiment_config()
        seed_everything(0, deterministic=True)
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            request = TrainRequest(data_root=Path("SMOKE_SYNTHETIC_NOT_BRATS"), run_dir=run_dir, config=config)
            train_loader, val_loader = smoke_loaders(config)
            ckpt = fit(
                request,
                train_loader,
                val_loader,
                smoke=True,
                model_cfg=smoke_model_cfg(config),
                epochs_override=1,
            )
            self.assertTrue((run_dir / "latest.pt").is_file())
            self.assertTrue((run_dir / "history.json").is_file())
            self.assertTrue(ckpt.is_file())
            history = json.loads((run_dir / "history.json").read_text(encoding="utf-8"))
            self.assertEqual(len(history), 1)
            self.assertIn("train_loss", history[0])
            self.assertIn("val_loss", history[0])
            self.assertIn("dice_wt", history[0])
            meta = json.loads((run_dir / "metadata.json").read_text(encoding="utf-8"))
            self.assertEqual(meta["status"], "smoke_synthetic_not_research")
            self.assertTrue(meta.get("smoke"))

            payload = load_checkpoint(run_dir / "latest.pt")
            self.assertTrue(payload.get("smoke"))
            from neurovisionx.models.factory import build_model

            model = build_model(smoke_model_cfg(config))
            restore_training_state(payload, model)
            import torch

            logits = model(torch.randn(1, 4, 16, 16, 16))
            self.assertEqual(tuple(logits.shape), (1, 4, 16, 16, 16))

    def test_train_script_smoke_flag_is_documented(self):
        from neurovisionx.utils.paths import project_root

        text = (project_root() / "scripts" / "train.py").read_text(encoding="utf-8")
        self.assertIn("--smoke", text)
        self.assertIn("Not BraTS", text)
