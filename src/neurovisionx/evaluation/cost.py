"""Parameter counts and wall-clock inference timing.

FLOPs estimates are not reported here. Only measured parameter counts and
timer results are recorded. Synthetic tensors in tests are not BraTS runtime.
"""

from __future__ import annotations

import time
from typing import Any

import torch
import torch.nn as nn


def count_parameters(model: nn.Module, *, trainable_only: bool = True) -> int:
    if trainable_only:
        return int(sum(p.numel() for p in model.parameters() if p.requires_grad))
    return int(sum(p.numel() for p in model.parameters()))


@torch.no_grad()
def time_inference(
    model: nn.Module,
    volume: torch.Tensor,
    *,
    n_warmup: int = 1,
    n_repeat: int = 3,
    mc_samples: int = 1,
    device: torch.device | None = None,
) -> dict[str, Any]:
    """Mean wall-clock milliseconds for one volume.

    ``mc_samples`` > 1 enables dropout and repeats the forward pass (Gal and
    Ghahramani, 2016). CUDA synchronization is used when the tensor is on GPU.
    """
    if device is None:
        device = volume.device
    model = model.to(device)
    volume = volume.to(device)
    if volume.ndim == 4:
        volume = volume.unsqueeze(0)
    model.eval()
    if mc_samples > 1:
        from neurovisionx.uncertainty.mc_dropout import enable_mc_dropout

        enable_mc_dropout(model)

    def _forward() -> None:
        if mc_samples <= 1:
            model.eval()
            _ = model(volume)
            return
        from neurovisionx.uncertainty.mc_dropout import mc_dropout_predict

        _ = mc_dropout_predict(model, volume, samples=mc_samples)

    for _ in range(max(n_warmup, 0)):
        _forward()
        if device.type == "cuda":
            torch.cuda.synchronize()

    elapsed: list[float] = []
    for _ in range(max(n_repeat, 1)):
        if device.type == "cuda":
            torch.cuda.synchronize()
        start = time.perf_counter()
        _forward()
        if device.type == "cuda":
            torch.cuda.synchronize()
        elapsed.append((time.perf_counter() - start) * 1000.0)

    arr = torch.tensor(elapsed)
    return {
        "mean_ms": float(arr.mean().item()),
        "std_ms": float(arr.std(unbiased=False).item()) if arr.numel() > 1 else 0.0,
        "n_repeat": int(n_repeat),
        "mc_samples": int(mc_samples),
        "device": str(device),
        "n_params": count_parameters(model),
        "note": "Wall-clock on the tensor that was passed. Not a clinical latency claim.",
    }


@torch.no_grad()
def measure_gpu_memory(
    model: nn.Module,
    volume: torch.Tensor,
    *,
    mc_samples: int = 1,
    device: torch.device | None = None,
) -> dict[str, Any]:
    """Peak CUDA allocator bytes during one forward. CPU returns a measured skip.

    This is `torch.cuda.max_memory_allocated`, not nvidia-smi reserved memory.
    Not a clinical hardware requirement.
    """
    if device is None:
        device = volume.device if volume.is_cuda else torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
    payload: dict[str, Any] = {
        "device": str(device),
        "peak_allocated_mb": None,
        "measured": False,
        "mc_samples": int(mc_samples),
        "note": "GPU memory not measured (CUDA unavailable or CPU tensor).",
    }
    if device.type != "cuda" or not torch.cuda.is_available():
        return payload
    model = model.to(device)
    volume = volume.to(device)
    if volume.ndim == 4:
        volume = volume.unsqueeze(0)
    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.synchronize(device)
    if mc_samples > 1:
        from neurovisionx.uncertainty.mc_dropout import mc_dropout_predict

        _ = mc_dropout_predict(model, volume, samples=mc_samples)
    else:
        model.eval()
        _ = model(volume)
    torch.cuda.synchronize(device)
    peak = float(torch.cuda.max_memory_allocated(device) / (1024.0 ** 2))
    payload.update(
        {
            "peak_allocated_mb": peak,
            "measured": True,
            "note": "Peak allocated MB from torch CUDA allocator during one forward.",
        }
    )
    return payload
