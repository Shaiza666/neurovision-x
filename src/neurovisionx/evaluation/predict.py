"""Checkpoint loading and volume inference.

Used after a real training job. Smoke checkpoints must not be evaluated as
BraTS results.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from neurovisionx.evaluation.inference import sliding_window_logits
from neurovisionx.models.factory import build_model
from neurovisionx.training.checkpoint import load_checkpoint


def load_model_from_checkpoint(
    path: Path,
    model_cfg: dict[str, Any],
    map_location: str | torch.device = "cpu",
) -> tuple[nn.Module, dict[str, Any]]:
    payload = load_checkpoint(path, map_location=map_location)
    if payload.get("smoke") or payload.get("extra", {}).get("smoke"):
        # Tag is also stored at top-level in save_checkpoint extra= update
        pass
    cfg = payload.get("config", {}).get("model", model_cfg) if isinstance(payload, dict) else model_cfg
    if not isinstance(cfg, dict) or "in_channels" not in cfg:
        cfg = model_cfg
    model = build_model(cfg)
    state = payload["model"] if isinstance(payload, dict) and "model" in payload else payload
    model.load_state_dict(state)
    model.eval()
    return model, payload if isinstance(payload, dict) else {}


def infer_volume(
    model: nn.Module,
    volume: torch.Tensor,
    patch_size: tuple[int, int, int],
    overlap: float = 0.5,
) -> torch.Tensor:
    """Return class map (D, H, W) from ``volume`` (1, C, D, H, W) or (C, D, H, W)."""
    if volume.ndim == 4:
        volume = volume.unsqueeze(0)
    logits = sliding_window_logits(model, volume, patch_size, overlap=overlap)
    return logits.argmax(dim=1)[0]
