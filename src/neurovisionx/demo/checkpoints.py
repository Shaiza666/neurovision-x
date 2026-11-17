"""Discover and load checkpoints that actually exist on disk.

Smoke weights are loadable for interface tests. They are not BraTS models.
A missing or incompatible file is an error, not a reason to invent logits.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch.nn as nn

from neurovisionx.experiments.catalog import catalog_experiments, experiments_root
from neurovisionx.models.factory import build_model
from neurovisionx.training.checkpoint import load_checkpoint
from neurovisionx.utils.config import load_experiment_config


def model_cfg_from_payload(payload: dict[str, Any], fallback: dict[str, Any] | None = None) -> dict[str, Any]:
    """Build the architecture that matches saved weights.

    Smoke jobs train a tiny U-Net (`training.smoke.features`) but store the
    full YAML under ``config.model``. Using the YAML widths would fail
    ``load_state_dict``. This helper prefers the smoke widths when tagged.
    """
    base = fallback if fallback is not None else load_experiment_config()["model"]
    cfg = dict(payload.get("config", {}).get("model") or base)
    if payload.get("smoke") or payload.get("extra", {}).get("smoke"):
        smoke = payload.get("config", {}).get("training", {}).get("smoke") or {}
        if "features" in smoke:
            cfg["features"] = list(smoke["features"])
        cfg["optional_monai"] = False
    return cfg


def patch_size_from_payload(payload: dict[str, Any]) -> tuple[int, int, int]:
    train = payload.get("config", {}).get("training") or {}
    if payload.get("smoke"):
        smoke = train.get("smoke") or {}
        if "patch_size" in smoke:
            return tuple(int(v) for v in smoke["patch_size"])
    schedule = train.get("schedule") or {}
    patch = schedule.get("patch_size", [96, 96, 96])
    return tuple(int(v) for v in patch)


def list_demo_checkpoints(root: Path | None = None) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for run in catalog_experiments(root or experiments_root()):
        run_path = Path(run["path"])
        for filename, kind in (("best.pt", "best"), ("latest.pt", "latest")):
            path = run_path / filename
            if not path.is_file():
                continue
            rows.append(
                {
                    "label": f"{run['run_name']} / {kind}"
                    + (" [SMOKE synthetic]" if run.get("smoke") else ""),
                    "path": path,
                    "run_name": run["run_name"],
                    "kind": kind,
                    "smoke": bool(run.get("smoke")),
                    "device_trained": run.get("device"),
                    "status": run.get("status"),
                }
            )
    return rows


def load_demo_model(path: Path, map_location: str = "cpu") -> tuple[nn.Module, dict[str, Any]]:
    payload = load_checkpoint(path, map_location=map_location)
    cfg = model_cfg_from_payload(payload)
    model = build_model(cfg)
    state = payload["model"] if "model" in payload else payload
    model.load_state_dict(state)
    model.eval()
    return model, payload
