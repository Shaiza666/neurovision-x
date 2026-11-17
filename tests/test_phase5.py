"""Phase 5 evaluation-framework tests. Synthetic tensors are not BraTS results."""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

from neurovisionx.evaluation.hard_cases import rank_failures
from neurovisionx.evaluation.metrics import brier_score_multiclass, precision_recall
from neurovisionx.evaluation.regions import region_scores
from neurovisionx.evaluation.summarize import format_summary, summarize_values
from neurovisionx.experiments.catalog import catalog_experiments
from neurovisionx.experiments.report import build_results_markdown, load_evaluation_config
from neurovisionx.utils.paths import project_root


class MetricExtensionTests(unittest.TestCase):
    def test_precision_recall_perfect_and_empty(self):
        mask = np.zeros((4, 4, 4), dtype=np.uint8)
        mask[1:3, 1:3, 1:3] = 1
        p, r = precision_recall(mask, mask)
        self.assertEqual(p, 1.0)
        self.assertEqual(r, 1.0)
        empty = np.zeros_like(mask)
        p0, r0 = precision_recall(empty, mask)
        self.assertEqual(p0, 0.0)
        self.assertEqual(r0, 0.0)

    def test_brier_zero_on_one_hot(self):
        probs = np.zeros((2, 3), dtype=np.float64)
        probs[1, :] = 1.0
        target = np.ones(3, dtype=np.int64)
        self.assertLess(brier_score_multiclass(probs, target), 1e-12)

    def test_region_scores_include_pr(self):
        volume = np.zeros((6, 6, 6), dtype=np.int64)
        volume[1:3, 1:3, 1:3] = 3
        scores = region_scores(volume, volume, compute_pr=True)
        self.assertEqual(scores["precision_et"], 1.0)
        self.assertEqual(scores["recall_et"], 1.0)

    def test_summarize_ci_and_missing(self):
        self.assertEqual(format_summary(summarize_values([])), "not measured")
        one = summarize_values([0.5])
        self.assertIsNone(one["ci_low"])
        self.assertEqual(one["n"], 1)
        many = summarize_values([0.2, 0.4, 0.6, 0.8])
        self.assertEqual(many["n"], 4)
        self.assertIsNotNone(many["ci_low"])
        self.assertLessEqual(many["ci_low"], many["mean"])
        self.assertGreaterEqual(many["ci_high"], many["mean"])
        self.assertIn("±", format_summary(many))

    def test_gpu_memory_not_invented_on_cpu(self):
        try:
            import torch
        except ImportError:
            self.skipTest("torch not installed")
        from neurovisionx.evaluation.cost import measure_gpu_memory
        from neurovisionx.models.factory import build_model

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
        volume = torch.randn(1, 4, 16, 16, 16)
        payload = measure_gpu_memory(model, volume, device=torch.device("cpu"))
        self.assertFalse(payload["measured"])
        self.assertIsNone(payload["peak_allocated_mb"])

    def test_rank_failures_worst_dice(self):
        rows = [
            {"patient_id": "a", "dice_et": 0.9},
            {"patient_id": "b", "dice_et": 0.1},
            {"patient_id": "c", "dice_et": 0.5},
        ]
        worst = rank_failures(rows, metric="dice_et", k=2, higher_is_better=True)
        self.assertEqual([row["patient_id"] for row in worst], ["b", "c"])


class ReportHonestyTests(unittest.TestCase):
    def test_evaluation_yaml_keeps_training_gate(self):
        spec = load_evaluation_config()
        self.assertTrue(spec["expensive_training_requires_approval"])
        self.assertEqual(spec["comparisons"]["A"]["run_id"], "A0_baseline")
        self.assertEqual(spec["comparisons"]["B"]["run_id"], "A2_attention")
        self.assertEqual(spec["comparisons"]["C"]["run_id"], "A5_combination")
        ids = {arm["id"] for arm in spec["component_ablations"]}
        self.assertEqual(ids, {"C_minus_se", "C_minus_boundary", "C_minus_mc"})

    def test_report_does_not_fill_abc_from_smoke(self):
        spec = load_evaluation_config()
        fake_smoke = [
            {
                "run_name": "smoke",
                "status": "smoke_synthetic_not_research",
                "smoke": True,
                "research_eligible": False,
                "device": "cpu",
                "cuda": False,
                "epochs_ran": 2,
                "history": [{"epoch": 1, "dice_et": 0.99, "train_loss": 0.1, "val_loss": 0.1}],
                "metrics_test": {
                    "n_patients": 1,
                    "summary": {"dice_et": {"mean": 0.99, "n": 1}},
                    "patients": [{"dice_et": 0.99}],
                },
            }
        ]
        md = build_results_markdown(fake_smoke, spec, figure_rels={})
        self.assertIn("not measured", md)
        self.assertIn("| Dice ET | not measured | not measured | not measured |", md)
        self.assertIn("are **not** BraTS", md)
        self.assertIn("not research", md.lower())
        self.assertIn("No improvement", md)

    def test_catalog_classifies_local_smoke_if_present(self):
        smoke = project_root() / "experiments" / "smoke"
        if not (smoke / "metadata.json").is_file():
            self.skipTest("no local smoke folder")
        runs = catalog_experiments()
        names = {row["run_name"]: row for row in runs}
        self.assertIn("smoke", names)
        self.assertTrue(names["smoke"]["smoke"])
        self.assertFalse(names["smoke"]["research_eligible"])

    def test_collect_results_writes_honest_docs(self):
        script = project_root() / "scripts" / "collect_results.py"
        completed = subprocess.run(
            [sys.executable, str(script)],
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertIn("not measured", completed.stdout.lower() + completed.stderr.lower())
        results = (project_root() / "docs" / "experimental_results.md").read_text(encoding="utf-8")
        summary = (project_root() / "docs" / "experiment_summary.md").read_text(encoding="utf-8")
        self.assertIn("not measured", results.lower())
        self.assertIn("not a clinically validated", results.lower())
        self.assertNotIn("proposed method is better", results.lower())
        self.assertIn("not compared on official data", summary.lower())

    def test_collect_results_does_not_require_brats(self):
        script = project_root() / "scripts" / "collect_results.py"
        with tempfile.TemporaryDirectory() as tmp:
            completed = subprocess.run(
                [
                    sys.executable,
                    str(script),
                    "--experiments-root",
                    tmp,
                    "--docs-root",
                    tmp,
                ],
                capture_output=True,
                text=True,
                check=False,
            )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        payload = json.loads(completed.stdout.split("Wrote")[0] or "{}")
        if "n_research_eligible" in payload:
            self.assertEqual(payload["n_research_eligible"], 0)


if __name__ == "__main__":
    unittest.main()
