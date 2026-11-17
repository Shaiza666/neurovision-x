"""Phase 2 CLI scripts exit honestly when official data is missing."""

from __future__ import annotations

import subprocess
import sys
import unittest

from neurovisionx.utils.paths import project_root


class Phase2ScriptTests(unittest.TestCase):
    def _run(self, name: str) -> subprocess.CompletedProcess[str]:
        script = project_root() / "scripts" / name
        return subprocess.run(
            [sys.executable, str(script)],
            capture_output=True,
            text=True,
            check=False,
        )

    def test_validate_exits_without_data(self):
        completed = self._run("validate_dataset.py")
        self.assertNotEqual(completed.returncode, 0)
        self.assertIn("No BraTS", completed.stdout + completed.stderr)

    def test_stats_exits_without_data(self):
        completed = self._run("dataset_stats.py")
        self.assertNotEqual(completed.returncode, 0)

    def test_qc_exits_without_data(self):
        completed = self._run("dataset_qc.py")
        self.assertNotEqual(completed.returncode, 0)
