"""Occlusion sensitivity for research visualization.

Maps produced here are not clinical explanations and must not be presented as
radiologist-facing evidence.
"""

from __future__ import annotations

import torch
import torch.nn as nn


@torch.no_grad()
def occlusion_sensitivity(
    model: nn.Module,
    volume: torch.Tensor,
    *,
    patch: int = 8,
    stride: int = 8,
    class_index: int = 2,
) -> torch.Tensor:
    """Drop cubic patches and record drop in predicted class mass.

    ``volume`` is (1, C, D, H, W). Returns a (D, H, W) sensitivity map.
    """
    model.eval()
    baseline = torch.softmax(model(volume), dim=1)[0, class_index].sum()
    _, _, depth, height, width = volume.shape
    sensitivity = torch.zeros(depth, height, width, device=volume.device)
    counts = torch.zeros_like(sensitivity)
    for z in range(0, depth, stride):
        for y in range(0, height, stride):
            for x in range(0, width, stride):
                occluded = volume.clone()
                occluded[
                    :,
                    :,
                    z : z + patch,
                    y : y + patch,
                    x : x + patch,
                ] = 0
                score = torch.softmax(model(occluded), dim=1)[0, class_index].sum()
                delta = (baseline - score).clamp(min=0)
                sensitivity[z : z + patch, y : y + patch, x : x + patch] += delta
                counts[z : z + patch, y : y + patch, x : x + patch] += 1
    counts = torch.clamp(counts, min=1)
    return sensitivity / counts
