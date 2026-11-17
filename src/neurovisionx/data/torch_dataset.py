"""PyTorch dataset over discovered BraTS patient cases."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from neurovisionx.data.inventory import PatientCase, discover_inventory
from neurovisionx.data.patches import extract_training_patch
from neurovisionx.data.preprocess import crop_or_pad, preprocess_patient_files, stack_modalities


class BraTSPatchDataset(Dataset):
    """Loads four MRI channels and a seg map, then extracts a 3D patch.

    Cases come from ``discover_inventory`` of the real tree. Missing files raise;
    they are not replaced with synthetic MRI.
    """

    def __init__(
        self,
        raw_root: Path,
        patient_ids: list[str] | tuple[str, ...],
        patch_size: tuple[int, int, int] = (96, 96, 96),
        *,
        random_crop: bool = True,
        seed: int = 2026,
        cases: dict[str, PatientCase] | None = None,
        target_spacing_mm: tuple[float, float, float] = (1.0, 1.0, 1.0),
        crop_foreground: bool = True,
        foreground_prob: float = 0.67,
    ) -> None:
        if not patient_ids:
            raise ValueError("patient_ids is empty")
        self.raw_root = Path(raw_root)
        self.patient_ids = tuple(patient_ids)
        self.patch_size = tuple(int(v) for v in patch_size)
        self.random_crop = random_crop
        self._rng = np.random.default_rng(seed)
        self.target_spacing_mm = target_spacing_mm
        self.crop_foreground = crop_foreground
        self.foreground_prob = foreground_prob
        if cases is None:
            inventory = discover_inventory(self.raw_root)
            cases = inventory.by_id()
        self._cases = cases

    def __len__(self) -> int:
        return len(self.patient_ids)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        patient_id = self.patient_ids[index]
        case = self._cases[patient_id]
        arrays = preprocess_patient_files(
            case.path_map(),
            target_spacing_mm=self.target_spacing_mm,
            crop_foreground=self.crop_foreground,
        )
        image = stack_modalities(arrays)
        seg = arrays["seg"]
        if self.random_crop:
            image, seg = extract_training_patch(
                image, seg, self.patch_size, self._rng, self.foreground_prob
            )
        else:
            image, seg = crop_or_pad(image, seg, self.patch_size, rng=None)
        return {
            "image": torch.from_numpy(np.ascontiguousarray(image)),
            "label": torch.from_numpy(np.ascontiguousarray(seg)),
            "patient_id": patient_id,
        }
