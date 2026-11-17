"""Sliding-window inference for 3D volumes.

Used at evaluation time on real NIfTI. Random tensors in tests are not MRI.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


@torch.no_grad()
def sliding_window_logits(
    model: nn.Module,
    volume: torch.Tensor,
    patch_size: tuple[int, int, int],
    overlap: float = 0.5,
) -> torch.Tensor:
    """``volume`` is (1, C, D, H, W). Returns logits of the same spatial size."""
    if volume.ndim != 5 or volume.shape[0] != 1:
        raise ValueError("volume must be (1, C, D, H, W)")
    model.eval()
    _, _, depth, height, width = volume.shape
    pz, py, px = patch_size
    step_z = max(int(pz * (1.0 - overlap)), 1)
    step_y = max(int(py * (1.0 - overlap)), 1)
    step_x = max(int(px * (1.0 - overlap)), 1)

    pad_d = max(pz - depth, 0)
    pad_h = max(py - height, 0)
    pad_w = max(px - width, 0)
    if pad_d or pad_h or pad_w:
        volume = F.pad(volume, (0, pad_w, 0, pad_h, 0, pad_d))
    _, _, padded_d, padded_h, padded_w = volume.shape

    acc = torch.zeros(1, 0, padded_d, padded_h, padded_w, device=volume.device)
    counts = torch.zeros(1, 1, padded_d, padded_h, padded_w, device=volume.device)
    initialized = False

    def _axis_starts(length: int, patch: int, step: int) -> list[int]:
        starts = list(range(0, length - patch + 1, step))
        last = length - patch
        if starts[-1] != last:
            starts.append(last)
        return starts

    for z in _axis_starts(padded_d, pz, step_z):
        for y in _axis_starts(padded_h, py, step_y):
            for x in _axis_starts(padded_w, px, step_x):
                patch = volume[:, :, z : z + pz, y : y + py, x : x + px]
                logits = model(patch)
                if not initialized:
                    acc = torch.zeros(
                        1, logits.shape[1], padded_d, padded_h, padded_w, device=volume.device
                    )
                    initialized = True
                acc[:, :, z : z + pz, y : y + py, x : x + px] += logits
                counts[:, :, z : z + pz, y : y + py, x : x + px] += 1

    counts = torch.clamp(counts, min=1)
    fused = acc / counts
    return fused[:, :, :depth, :height, :width]
