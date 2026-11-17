"""Mixed-precision helpers. AMP is used on CUDA when enabled; CPU stays fp32."""

from __future__ import annotations

from contextlib import nullcontext
from typing import Any

import torch


def amp_enabled(train_cfg: dict[str, Any], device: torch.device) -> bool:
    return bool(train_cfg.get("schedule", {}).get("amp", False)) and device.type == "cuda"


def make_scaler(enabled: bool) -> Any:
    try:
        return torch.amp.GradScaler("cuda", enabled=enabled)
    except (TypeError, AttributeError):
        return torch.cuda.amp.GradScaler(enabled=enabled)


def autocast(device: torch.device, enabled: bool):
    if not enabled:
        return nullcontext()
    try:
        return torch.amp.autocast(device_type=device.type, enabled=True)
    except (TypeError, AttributeError):
        return torch.cuda.amp.autocast(enabled=True)
