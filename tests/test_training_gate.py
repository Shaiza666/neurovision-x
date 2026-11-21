"""Training and eval refuse to run without official BraTS folders."""

from __future__ import annotations

import subprocess
import sys
import unittest
from pathlib import Path

from neurovisionx.data.dataset import MissingBraTSDataError
from neurovisionx.training.loop import refuse_without_real_data
from neurovisionx.utils.paths import project_root


class TrainingGateTests(unittest.TestCase):
    def test_refuse_without_real_data(self):
        missing = project_root() / "data" / "raw" / "brats"
        with self.assertRaises(MissingBraTSDataError):
            refuse_without_real_data(missing)

    def test_train_script_exits_without_data(self):
        script = project_root() / "scripts" / "train.py"
        completed = subprocess.run(
            [sys.executable, str(script)],
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertNotEqual(completed.returncode, 0)
        combined = completed.stdout + completed.stderr
        self.assertIn("No BraTS volumes", combined)
        self.assertNotIn("Dice", combined)

    def test_infer_script_refuses_smoke_without_flag(self):
        ckpt = project_root() / "experiments" / "smoke" / "best.pt"
        if not ckpt.is_file():
            self.skipTest("smoke checkpoint not on disk")
        script = project_root() / "scripts" / "infer.py"
        completed = subprocess.run(
            [sys.executable, str(script), "--run-name", "smoke"],
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertNotEqual(completed.returncode, 0)
        self.assertIn("Smoke checkpoint refused", completed.stdout + completed.stderr)

    def test_infer_script_missing_case_dir_is_explicit(self):
        ckpt = project_root() / "experiments" / "smoke" / "best.pt"
        if not ckpt.is_file():
            self.skipTest("smoke checkpoint not on disk")
        script = project_root() / "scripts" / "infer.py"
        missing = project_root() / "data" / "raw" / "brats" / "does_not_exist"
        completed = subprocess.run(
            [
                sys.executable,
                str(script),
                "--run-name",
                "smoke",
                "--allow-smoke",
                "--case-dir",
                str(missing),
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        combined = completed.stdout + completed.stderr
        self.assertNotEqual(completed.returncode, 0)
        self.assertIn("not synthesized", combined.lower())

    def test_infer_script_smoke_shape_check(self):
        ckpt = project_root() / "experiments" / "smoke" / "best.pt"
        if not ckpt.is_file():
            self.skipTest("smoke checkpoint not on disk")
        script = project_root() / "scripts" / "infer.py"
        completed = subprocess.run(
            [sys.executable, str(script), "--run-name", "smoke", "--allow-smoke"],
            capture_output=True,
            text=True,
            check=False,
        )
        combined = completed.stdout + completed.stderr
        self.assertEqual(completed.returncode, 0, combined)
        self.assertIn("SMOKE shape check", combined)
        self.assertIn("not research", combined)

    def test_evaluate_script_exits_without_data(self):
        script = project_root() / "scripts" / "evaluate.py"
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
