"""Checkpoint fetch: copy or download, then SHA-256. No network in these tests."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from neurovisionx.demo.fetch_weights import CheckpointFetchError, file_sha256, materialize_checkpoint
from neurovisionx.utils.config import load_yaml
from neurovisionx.utils.paths import project_root


class FetchWeightTests(unittest.TestCase):
    def test_copy_verifies_sha256(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "src.pt"
            source.write_bytes(b"checkpoint-bytes")
            digest = file_sha256(source)
            target = root / "experiments" / "A0_baseline" / "best.pt"
            kind = materialize_checkpoint(
                {"path": "src.pt", "sha256": digest, "filename": "best.pt"},
                target,
                repo_root=root,
            )
            self.assertEqual(kind, "copied")
            self.assertEqual(target.read_bytes(), b"checkpoint-bytes")
            again = materialize_checkpoint(
                {"path": "src.pt", "sha256": digest, "filename": "best.pt"},
                target,
                repo_root=root,
            )
            self.assertEqual(again, "present")

    def test_mismatch_deletes_the_destination(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "src.pt"
            source.write_bytes(b"not-the-expected-bytes")
            target = root / "experiments" / "A2_attention" / "best.pt"
            with self.assertRaises(CheckpointFetchError):
                materialize_checkpoint(
                    {"path": "src.pt", "sha256": "0" * 64, "filename": "best.pt"},
                    target,
                    repo_root=root,
                )
            self.assertFalse(target.exists())

    def test_download_url_is_hashed(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "blob.pt"
            source.write_bytes(b"from-url")
            digest = file_sha256(source)
            target = root / "out" / "best.pt"
            kind = materialize_checkpoint(
                {"url": source.resolve().as_uri(), "sha256": digest, "filename": "best.pt"},
                target,
                repo_root=root,
            )
            self.assertEqual(kind, "downloaded")
            self.assertEqual(file_sha256(target), digest)

    def test_published_yaml_points_at_the_release(self):
        payload = load_yaml(project_root() / "configs" / "checkpoints.yaml")
        self.assertTrue(payload["published"])
        self.assertEqual(payload["tag"], "v0.1-msd-cpu")
        self.assertIn("not a clinically validated", payload["clinical_disclaimer"].lower())
        expected = {
            "A0_baseline": (
                "https://github.com/Shaiza666/neurovision-x/releases/download/v0.1-msd-cpu/A0_baseline_best.pt",
                "f9fea768469d3e6b21c8535bbf180a7f2db42fb14f44c9c22d02a24c6e80c172",
                16999253,
            ),
            "A2_attention": (
                "https://github.com/Shaiza666/neurovision-x/releases/download/v0.1-msd-cpu/A2_attention_best.pt",
                "6883c13894b6591fd8c46e7c95e730851fbe249327db3d7093edca6a2d89d1a3",
                17086149,
            ),
            "A5_combination": (
                "https://github.com/Shaiza666/neurovision-x/releases/download/v0.1-msd-cpu/A5_combination_best.pt",
                "4be1e938f2d2e816a938f7a54a8f43c68cfdf4c43a3baaef03b9bdbffddd1c11",
                17004677,
            ),
        }
        self.assertEqual(set(payload["runs"]), set(expected))
        for run, (url, digest, nbytes) in expected.items():
            spec = payload["runs"][run]
            self.assertEqual(spec["url"], url)
            self.assertEqual(spec["sha256"], digest)
            self.assertEqual(spec["bytes"], nbytes)
            self.assertEqual(spec["filename"], "best.pt")


if __name__ == "__main__":
    unittest.main()
