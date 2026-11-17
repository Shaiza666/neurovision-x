"""Config loading tests."""

from __future__ import annotations

import unittest

from neurovisionx.utils.config import load_experiment_config


class ConfigTests(unittest.TestCase):
    def test_loads_repo_yaml(self):
        cfg = load_experiment_config()
        self.assertEqual(cfg["data"]["splits"]["strategy"], "patient_level")
        self.assertEqual(cfg["model"]["name"], "unet3d")
        self.assertEqual(cfg["training"]["seed"], 2026)
        self.assertTrue(cfg["training"]["logging"]["require_real_data"])
        ratios = cfg["data"]["splits"]
        total = ratios["train_ratio"] + ratios["val_ratio"] + ratios["test_ratio"]
        self.assertAlmostEqual(total, 1.0)


if __name__ == "__main__":
    unittest.main()
