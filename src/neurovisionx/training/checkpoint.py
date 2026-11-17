"""Checkpoint save/load. Recovery restores model, optimizer, scaler, and epoch."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import torch.nn as nn


def save_checkpoint(
    path: Path,
    *,
    model: nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    scaler: Any | None = None,
    epoch: int,
    history: list[dict[str, float]],
    config: dict[str, Any],
    best_metric: float,
    extra: dict[str, Any] | None = None,
) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "model": model.state_dict(),
        "epoch": epoch,
        "history": history,
        "config": config,
        "best_metric": best_metric,
        "clinical_disclaimer": (
            "Research prototype. Not a clinically validated diagnostic tool."
        ),
    }
    if optimizer is not None:
        payload["optimizer"] = optimizer.state_dict()
    if scaler is not None:
        payload["scaler"] = scaler.state_dict()
    if extra:
        payload.update(extra)
    torch.save(payload, path)
    return path


def load_checkpoint(path: Path, map_location: str | torch.device = "cpu") -> dict[str, Any]:
    if not Path(path).is_file():
        raise FileNotFoundError(f"No checkpoint at {path}")
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def restore_training_state(
    payload: dict[str, Any],
    model: nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    scaler: Any | None = None,
) -> int:
    """Load weights and optional optimizer/scaler. Returns the next epoch index."""
    state = payload["model"] if "model" in payload else payload
    model.load_state_dict(state)
    if optimizer is not None and "optimizer" in payload:
        optimizer.load_state_dict(payload["optimizer"])
    if scaler is not None and "scaler" in payload:
        scaler.load_state_dict(payload["scaler"])
    return int(payload.get("epoch", 0)) + 1
