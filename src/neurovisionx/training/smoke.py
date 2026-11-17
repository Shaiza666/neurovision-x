"""Synthetic smoke training. Not BraTS. Not research results."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader, Dataset

from neurovisionx.models.factory import build_model
from neurovisionx.training.loop import TrainRequest
from neurovisionx.utils.config import load_experiment_config
from neurovisionx.utils.metadata import build_run_metadata, write_metadata
from neurovisionx.utils.paths import project_root
from neurovisionx.utils.seed import seed_everything

SMOKE_NOTE = (
    "SMOKE TEST: random tensors only. Not MRI, not BraTS, not research results. "
    "Do not cite these losses or Dice numbers."
)


class SyntheticSmokeDataset(Dataset):
    """Tiny random volumes with a cubic blob. Not medical data."""

    def __init__(self, n: int, shape: tuple[int, int, int], n_channels: int, seed: int) -> None:
        self.n = n
        g = torch.Generator().manual_seed(seed)
        self.images = torch.randn(n, n_channels, *shape, generator=g)
        labels = torch.zeros(n, *shape, dtype=torch.long)
        d, h, w = shape
        labels[:, d // 4 : d // 2, h // 4 : h // 2, w // 4 : w // 2] = 1
        labels[:, d // 3 : d // 2, h // 3 : h // 2, w // 3 : w // 2] = 2
        labels[:, d // 3 + 1 : d // 2 - 1, h // 3 + 1 : h // 2 - 1, w // 3 + 1 : w // 2 - 1] = 3
        self.labels = labels

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, index: int) -> dict[str, Any]:
        return {
            "image": self.images[index],
            "label": self.labels[index],
            "patient_id": f"smoke_{index:02d}",
        }


def prepare_smoke_request(run_name: str = "smoke") -> TrainRequest:
    root = project_root()
    config = load_experiment_config()
    seed_everything(int(config["training"]["seed"]), bool(config["training"].get("deterministic", True)))
    run_dir = root / config["data"]["paths"]["output_root"] / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    write_metadata(
        run_dir / "metadata.json",
        build_run_metadata(config, {"run_name": run_name, "status": "smoke_synthetic_not_research", "note": SMOKE_NOTE}),
    )
    return TrainRequest(data_root=Path("SMOKE_SYNTHETIC_NOT_BRATS"), run_dir=run_dir, config=config)


def smoke_loaders(config: dict[str, Any]) -> tuple[DataLoader, DataLoader]:
    smoke_cfg = config["training"].get("smoke", {})
    shape = tuple(int(v) for v in smoke_cfg.get("patch_size", [16, 16, 16]))
    n_train = int(smoke_cfg.get("n_train", 4))
    n_val = int(smoke_cfg.get("n_val", 2))
    seed = int(config["training"]["seed"])
    channels = int(config["model"].get("in_channels", 4))
    train_ds = SyntheticSmokeDataset(n_train, shape, channels, seed)
    val_ds = SyntheticSmokeDataset(n_val, shape, channels, seed + 1)
    batch = int(smoke_cfg.get("batch_size", 1))
    return (
        DataLoader(train_ds, batch_size=batch, shuffle=True),
        DataLoader(val_ds, batch_size=1, shuffle=False),
    )


def smoke_model_cfg(config: dict[str, Any]) -> dict[str, Any]:
    cfg = dict(config["model"])
    smoke_cfg = config["training"].get("smoke", {})
    if "features" in smoke_cfg:
        cfg["features"] = smoke_cfg["features"]
    else:
        cfg["features"] = [8, 16]
    cfg["optional_monai"] = False
    return cfg
