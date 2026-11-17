"""Phase 4 unit tests on synthetic tensors. Not BraTS. Not research results."""

from __future__ import annotations

import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

from neurovisionx.experiments.ablation import apply_ablation, format_plan_table, list_ablation_runs
from neurovisionx.evaluation.hard_cases import (
    binary_auprc,
    et_voxel_count,
    size_stratum,
    spearman_uncertainty_dice,
    uncertainty_error_report,
    worst_quartile_labels,
)
from neurovisionx.evaluation.metrics import expected_calibration_error, negative_log_likelihood
from neurovisionx.utils.paths import project_root


class AblationPlanTests(unittest.TestCase):
    def test_six_independent_arms(self):
        runs = {run.id: run for run in list_ablation_runs()}
        self.assertTrue(
            {
                "A0_baseline",
                "A1_se_fusion",
                "A2_attention",
                "A3_boundary_et",
                "A4_mc_on_A0",
                "A5_combination",
            }.issubset(set(runs))
        )
        self.assertTrue(runs["A0_baseline"].train)
        self.assertFalse(runs["A4_mc_on_A0"].train)
        self.assertEqual(runs["A4_mc_on_A0"].checkpoint_from, "A0_baseline")
        self.assertEqual(runs["A3_boundary_et"].spec["loss"]["boundary_classes"], [3])
        self.assertEqual(runs["A0_baseline"].spec["loss"]["boundary_weight"], 0.0)

    def test_overrides_do_not_change_baseline_yaml_defaults(self):
        from neurovisionx.utils.config import load_experiment_config

        base = load_experiment_config()
        self.assertEqual(base["model"].get("fusion", "early"), "early")
        self.assertEqual(base["training"]["loss"]["boundary_weight"], 0.0)
        run, merged = apply_ablation("A3_boundary_et", base)
        self.assertEqual(run.isolates, "boundary")
        self.assertGreater(merged["training"]["loss"]["boundary_weight"], 0.0)
        self.assertEqual(base["training"]["loss"]["boundary_weight"], 0.0)

    def test_plan_script_prints_and_does_not_train(self):
        script = project_root() / "scripts" / "ablation_plan.py"
        completed = subprocess.run(
            [sys.executable, str(script)],
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertIn("A0_baseline", completed.stdout)
        self.assertIn("did not launch training", completed.stdout.lower())
        table = format_plan_table()
        self.assertIn("no training started", table.lower())


class BoundaryClassTests(unittest.TestCase):
    def test_et_only_differs_from_all_foreground(self):
        try:
            import torch
        except ImportError:
            self.skipTest("torch not installed")
        from neurovisionx.losses.segmentation import CombinedSegLoss, boundary_loss

        probs = torch.zeros(1, 4, 4, 4, 4)
        probs[:, 3] = 1.0
        dist = torch.zeros(1, 4, 4, 4, 4)
        dist[:, 1] = 10.0
        dist[:, 3] = 2.0
        all_fg = float(boundary_loss(probs, dist, classes=None))
        et_only = float(boundary_loss(probs, dist, classes=[3]))
        self.assertNotAlmostEqual(all_fg, et_only, places=5)
        self.assertAlmostEqual(et_only, 2.0, places=5)

        logits = torch.zeros(1, 4, 4, 4, 4)
        logits[:, 3] = 8.0
        target = torch.zeros(1, 4, 4, 4, dtype=torch.long)
        target[:, 1, 1, 1] = 3
        combined = CombinedSegLoss(
            dice_weight=0.0,
            ce_weight=0.0,
            boundary_weight=1.0,
            boundary_classes=[3],
        )
        loss = float(combined(logits, target, distance_map=dist))
        self.assertGreater(loss, 0.0)


class CalibrationAndHardCaseTests(unittest.TestCase):
    def test_nll_near_zero_when_confident_and_correct(self):
        probs = np.zeros((2, 4), dtype=np.float64)
        probs[1, :] = 1.0
        target = np.ones(4, dtype=np.int64)
        self.assertLess(negative_log_likelihood(probs, target), 1e-6)

    def test_auprc_and_stratum(self):
        scores = np.array([0.9, 0.8, 0.1, 0.0])
        labels = np.array([1, 1, 0, 0])
        self.assertGreater(binary_auprc(scores, labels), 0.9)
        counts = np.array([1, 2, 10, 12], dtype=np.float64)
        dice = np.array([0.2, 0.3, 0.8, 0.9])
        stratum = size_stratum(counts, dice, q=0.25)
        self.assertEqual(stratum["n_small"], 1)
        self.assertLess(stratum["small_mean"], stratum["rest_mean"])
        self.assertEqual(int(worst_quartile_labels(dice).sum()), 1)

    def test_spearman_negative_when_uncertainty_tracks_error(self):
        try:
            import scipy  # noqa: F401
        except ImportError:
            self.skipTest("scipy not installed")
        uncertainty = np.linspace(0.1, 1.0, 12)
        dice = 1.0 - uncertainty
        stats = spearman_uncertainty_dice(uncertainty, dice)
        self.assertLess(stats["rho"], -0.9)
        report = uncertainty_error_report(uncertainty, dice, np.arange(12, dtype=np.float64))
        self.assertLess(report["spearman"]["rho"], 0.0)

    def test_et_voxel_count(self):
        label = np.zeros((4, 4, 4), dtype=np.uint8)
        label[0, 0, 0] = 4  # legacy ET
        label[0, 0, 1] = 3
        self.assertEqual(et_voxel_count(label), 2)

    def test_ece_import_still_works(self):
        confidence = np.array([0.0, 1.0])
        correct = np.array([0.0, 1.0])
        self.assertLess(expected_calibration_error(confidence, correct, n_bins=2), 0.05)


class McDropoutAndCostTests(unittest.TestCase):
    def test_mc_stats_and_timing(self):
        try:
            import torch
        except ImportError:
            self.skipTest("torch not installed")
        from neurovisionx.evaluation.cost import count_parameters, time_inference
        from neurovisionx.models.factory import build_model
        from neurovisionx.uncertainty.mc_dropout import mc_dropout_stats

        model = build_model(
            {
                "name": "unet3d",
                "in_channels": 4,
                "out_channels": 4,
                "features": [8, 16],
                "dropout": 0.2,
                "optional_monai": False,
                "fusion": "early",
            }
        )
        volume = torch.randn(1, 4, 16, 16, 16)
        stats = mc_dropout_stats(model, volume, samples=3)
        self.assertEqual(tuple(stats["mean_probs"].shape), (4, 16, 16, 16))
        self.assertEqual(tuple(stats["entropy"].shape), (16, 16, 16))
        self.assertTrue(torch.all(stats["mutual_info"] >= -1e-5))
        self.assertGreater(count_parameters(model), 0)
        timed = time_inference(model, volume, n_warmup=0, n_repeat=1, mc_samples=2)
        self.assertIn("mean_ms", timed)
        self.assertGreaterEqual(timed["mean_ms"], 0.0)

    def test_overlay_writes_png(self):
        try:
            import matplotlib  # noqa: F401
        except ImportError:
            self.skipTest("matplotlib not installed")
        from neurovisionx.uncertainty.visualize import save_uncertainty_overlay

        image = np.zeros((8, 8))
        pred = np.zeros((8, 8), dtype=np.uint8)
        entropy = np.linspace(0, 1, 64).reshape(8, 8)
        error = np.zeros((8, 8), dtype=bool)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "overlay.png"
            save_uncertainty_overlay(image, pred, entropy, error, path)
            self.assertTrue(path.is_file())
            self.assertGreater(path.stat().st_size, 0)


class ReliabilityScriptGateTests(unittest.TestCase):
    def test_train_ablation_a4_is_eval_only(self):
        script = project_root() / "scripts" / "train.py"
        completed = subprocess.run(
            [sys.executable, str(script), "--ablation", "A4_mc_on_A0"],
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertNotEqual(completed.returncode, 0)
        combined = completed.stdout + completed.stderr
        self.assertIn("evaluation-only", combined)

    def test_analyze_reliability_exits_without_data(self):
        script = project_root() / "scripts" / "analyze_reliability.py"
        completed = subprocess.run(
            [sys.executable, str(script)],
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertNotEqual(completed.returncode, 0)
        combined = completed.stdout + completed.stderr
        self.assertIn("No BraTS volumes", combined)


if __name__ == "__main__":
    unittest.main()
