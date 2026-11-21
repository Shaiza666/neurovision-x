"""MSD Task01 mapping. Fixtures are synthetic, not Decathlon voxels."""

from __future__ import annotations

import unittest
from pathlib import Path

import numpy as np

from neurovisionx.data.inventory import PatientCase, discover_inventory
from neurovisionx.data.msd import (
    MSDSchemaError,
    assert_task01_labels,
    fit_to_grid,
    modality_index,
    remap_msd_labels,
    split_msd_image,
    subset_split,
    write_brats_style_case,
)
from neurovisionx.data.splits import split_from_cases, split_patients
from neurovisionx.utils.config import load_yaml
from neurovisionx.utils.paths import project_root
from tests.nifti_fixtures import write_nifti


TASK01_MODALITY = {"0": "FLAIR", "1": "T1w", "2": "T1gd", "3": "T2w"}
TASK01_LABELS = {
    "0": "background",
    "1": "edema",
    "2": "non-enhancing tumor",
    "3": "enhancing tumour",
}


class MSDMappingTests(unittest.TestCase):
    def test_modality_index_is_not_brats_channel_order(self):
        index = modality_index(TASK01_MODALITY)
        self.assertEqual(index, {"flair": 0, "t1": 1, "t1ce": 2, "t2": 3})

    def test_unknown_modality_is_rejected(self):
        with self.assertRaises(MSDSchemaError):
            modality_index({"0": "FLAIR", "1": "T1w", "2": "T1gd", "3": "ADC"})

    def test_label_schema_accepts_decathlon_names(self):
        assert_task01_labels(TASK01_LABELS)

    def test_label_schema_rejects_brats_order(self):
        swapped = {
            "0": "background",
            "1": "necrotic",
            "2": "edema",
            "3": "enhancing tumour",
        }
        with self.assertRaises(MSDSchemaError):
            assert_task01_labels(swapped)

    def test_remap_swaps_edema_and_nonenhancing(self):
        seg = np.array([0, 1, 2, 3], dtype=np.uint8)
        remapped = remap_msd_labels(seg)
        self.assertEqual(remapped.tolist(), [0, 2, 1, 3])

    def test_remap_rejects_legacy_brats_4(self):
        with self.assertRaises(MSDSchemaError):
            remap_msd_labels(np.array([0, 4], dtype=np.int16))

    def test_split_channel_last(self):
        image = np.zeros((2, 2, 2, 4), dtype=np.float32)
        image[..., 0] = 10
        image[..., 1] = 20
        image[..., 2] = 30
        image[..., 3] = 40
        channels = split_msd_image(image, modality_index(TASK01_MODALITY))
        self.assertEqual(float(channels["flair"].mean()), 10.0)
        self.assertEqual(float(channels["t1"].mean()), 20.0)
        self.assertEqual(float(channels["t1ce"].mean()), 30.0)
        self.assertEqual(float(channels["t2"].mean()), 40.0)

    def test_converted_folder_is_a_complete_case(self):
        import tempfile

        index = modality_index(TASK01_MODALITY)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            image = np.zeros((8, 8, 8, 4), dtype=np.float32)
            image[..., 0] = 3
            image[..., 1] = 4
            image[..., 2] = 5
            image[..., 3] = 6
            image[2:6, 2:6, 2:6, :] += 8
            seg = np.zeros((8, 8, 8), dtype=np.uint8)
            seg[3:6, 3:6, 3:6] = 1
            seg[4:6, 4:6, 4:6] = 2
            seg[4:5, 4:5, 4:5] = 3
            image_path = root / "imagesTr" / "BRATS_900.nii.gz"
            label_path = root / "labelsTr" / "BRATS_900.nii.gz"
            write_nifti(image_path, image)
            write_nifti(label_path, seg)
            dest = root / "out"
            row = write_brats_style_case(image_path, label_path, dest, index)
            self.assertEqual(row["patient_id"], "BRATS_900")
            inventory = discover_inventory(dest)
            self.assertEqual(inventory.n_complete, 1)
            case = inventory.cases[0]
            from neurovisionx.data.io import load_nifti

            flair = load_nifti(case.path_map()["flair"])
            t1 = load_nifti(case.path_map()["t1"])
            t1ce = load_nifti(case.path_map()["t1ce"])
            t2 = load_nifti(case.path_map()["t2"])
            labels = load_nifti(case.path_map()["seg"])
            self.assertAlmostEqual(float(np.median(flair[2:6, 2:6, 2:6])), 11.0, places=4)
            self.assertAlmostEqual(float(np.median(t1[2:6, 2:6, 2:6])), 12.0, places=4)
            self.assertAlmostEqual(float(np.median(t1ce[2:6, 2:6, 2:6])), 13.0, places=4)
            self.assertAlmostEqual(float(np.median(t2[2:6, 2:6, 2:6])), 14.0, places=4)
            # MSD 1 edema → 2, MSD 2 non-enhancing → 1, MSD 3 enhancing → 3
            self.assertIn(2, set(int(v) for v in np.unique(labels)))
            self.assertIn(1, set(int(v) for v in np.unique(labels)))
            self.assertIn(3, set(int(v) for v in np.unique(labels)))
            self.assertNotIn(4, set(int(v) for v in np.unique(labels)))

    def test_fit_to_grid_keeps_centered_tumor_and_pads(self):
        image = np.zeros((1, 4, 4, 4), dtype=np.float32)
        seg = np.zeros((4, 4, 4), dtype=np.int16)
        seg[1, 1, 1] = 3
        image[0, 1, 1, 1] = 9
        fitted, fitted_seg, retained = fit_to_grid(image, seg, (8, 8, 8))
        self.assertEqual(fitted.shape, (1, 8, 8, 8))
        self.assertEqual(int((fitted_seg == 3).sum()), 1)
        self.assertEqual(retained["et"]["fraction"], 1.0)

    def test_fit_to_grid_reports_clipped_labels(self):
        seg = np.zeros((6, 6, 6), dtype=np.int16)
        seg[0, 0, 0] = 3
        seg[3, 3, 3] = 3
        image = np.ones((1, 6, 6, 6), dtype=np.float32)
        _, fitted_seg, retained = fit_to_grid(image, seg, (2, 2, 2), center=np.array([0.0, 0.0, 0.0]))
        self.assertEqual(int((fitted_seg == 3).sum()), 1)
        self.assertEqual(retained["et"]["fraction"], 0.5)

    def test_subset_split_does_not_leak(self):
        full = split_patients([f"p{i:03d}" for i in range(30)], seed=2026)
        subset = subset_split(full, {"train": 4, "val": 2, "test": 3})
        self.assertEqual(subset.train, full.train[:4])
        self.assertEqual(subset.val, full.val[:2])
        self.assertEqual(subset.test, full.test[:3])
        self.assertTrue(set(subset.train).isdisjoint(subset.val))
        self.assertTrue(set(subset.train).isdisjoint(subset.test))

    def test_official_subfolders_are_not_reshuffled(self):
        cases = []
        for split_name, ids in (
            ("training", ("a", "b", "c")),
            ("validation", ("d",)),
            ("test", ("e",)),
        ):
            for patient_id in ids:
                cases.append(
                    PatientCase(
                        patient_id=f"{split_name}/{patient_id}",
                        folder="unused",
                        files={"t1": "t", "t1ce": "c", "t2": "2", "flair": "f", "seg": "s"},
                        missing=(),
                        naming="brats2021",
                        source_subdir=split_name,
                        unlabeled=False,
                    )
                )
        split = split_from_cases(cases, seed=2026, honor_official_validation=True)
        self.assertEqual(set(split.train), {"training/a", "training/b", "training/c"})
        self.assertEqual(set(split.val), {"validation/d"})
        self.assertEqual(set(split.test), {"test/e"})

    def test_default_data_yaml_still_points_at_brats(self):
        data = load_yaml(project_root() / "configs" / "data.yaml")
        self.assertEqual(data["paths"]["raw_root"], "data/raw/brats")
        msd = load_yaml(project_root() / "configs" / "data_msd.yaml")
        self.assertEqual(msd["paths"]["raw_root"], "data/raw/msd_task01")
        cpu = load_yaml(project_root() / "configs" / "data_msd_cpu.yaml")
        self.assertEqual(cpu["preprocessing"]["target_spacing_mm"], [2.0, 2.0, 2.0])
        self.assertFalse(cpu["preprocessing"]["crop_foreground"])


if __name__ == "__main__":
    unittest.main()
