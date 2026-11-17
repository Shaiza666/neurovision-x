"""Patient-level split leakage tests."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from neurovisionx.data.splits import assert_no_leakage, load_split, save_split, split_patients


class SplitTests(unittest.TestCase):
    def test_no_overlap_and_covers_all(self):
        ids = [f"p{i:02d}" for i in range(20)]
        split = split_patients(ids, seed=2026)
        assert_no_leakage(split)
        self.assertEqual(set(split.all_ids()), set(ids))
        self.assertGreaterEqual(len(split.train), 1)
        self.assertGreaterEqual(len(split.val), 1)
        self.assertGreaterEqual(len(split.test), 1)

    def test_seed_is_deterministic(self):
        ids = [f"p{i:02d}" for i in range(12)]
        a = split_patients(ids, seed=7)
        b = split_patients(ids, seed=7)
        self.assertEqual(a, b)

    def test_duplicate_ids_rejected(self):
        with self.assertRaises(ValueError):
            split_patients(["a", "a", "b", "c"])

    def test_official_validation_is_held_out(self):
        from types import SimpleNamespace

        from neurovisionx.data.splits import split_from_cases

        def case(patient_id: str, subdir: str | None) -> SimpleNamespace:
            return SimpleNamespace(
                patient_id=patient_id,
                source_subdir=subdir,
                is_complete=lambda: True,
            )

        cases = [case(f"train/p{i:02d}", "training") for i in range(8)]
        cases.append(case("validation/hold", "validation"))
        split = split_from_cases(cases, seed=2026, honor_official_validation=True)
        self.assertIn("validation/hold", split.val)
        self.assertNotIn("validation/hold", split.train)
        self.assertNotIn("validation/hold", split.test)

    def test_save_load_roundtrip(self):
        ids = [f"p{i:02d}" for i in range(10)]
        split = split_patients(ids, seed=3)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "patients.json"
            save_split(split, path)
            loaded = load_split(path)
        self.assertEqual(loaded, split)

    def test_ratios_must_sum_to_one(self):
        with self.assertRaises(ValueError):
            split_patients(["a", "b", "c", "d"], train_ratio=0.5, val_ratio=0.5, test_ratio=0.5)


if __name__ == "__main__":
    unittest.main()
