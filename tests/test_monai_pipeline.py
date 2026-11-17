"""MONAI transform construction. Skips if MONAI is not installed."""

from __future__ import annotations

import unittest


class MonaiPipelineTests(unittest.TestCase):
    def test_compose_builds(self):
        try:
            import monai  # noqa: F401
        except ImportError:
            self.skipTest("monai not installed")
        from neurovisionx.data.monai_pipeline import build_monai_transforms

        transform = build_monai_transforms(training=True, patch_size=(16, 16, 16))
        self.assertIsNotNone(transform)

    def test_missing_monai_is_explicit(self):
        import importlib.util

        from neurovisionx.data import monai_pipeline

        if importlib.util.find_spec("monai") is not None:
            self.skipTest("monai is installed; cannot test missing-import path")
        with self.assertRaises(ImportError):
            monai_pipeline.require_monai()
