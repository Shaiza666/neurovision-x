"""Segmentation losses.

Dice (Milletari et al., 2016) is a regional overlap objective:

    Dice_c = 2 <p, g> / (|p|_1 + |g|_1)

Boundary loss (Kervadec et al., 2021) multiplies softmax probabilities by a
precomputed signed distance map of the ground-truth contour. Distance maps
must come from real labels, not fabricated volumes.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def dice_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    *,
    smooth: float = 1e-5,
    include_background: bool = False,
) -> torch.Tensor:
    """Soft Dice loss. ``target`` is integer labels (N, D, H, W) or one-hot."""
    n_classes = logits.shape[1]
    probs = torch.softmax(logits, dim=1)
    if target.ndim == logits.ndim:
        one_hot = target
    else:
        one_hot = F.one_hot(target.long(), n_classes).permute(0, 4, 1, 2, 3).float()
    dims = (0, 2, 3, 4)
    start = 0 if include_background else 1
    intersection = (probs[:, start:] * one_hot[:, start:]).sum(dim=dims)
    union = probs[:, start:].sum(dim=dims) + one_hot[:, start:].sum(dim=dims)
    dice = (2 * intersection + smooth) / (union + smooth)
    return 1.0 - dice.mean()


def boundary_loss(probs: torch.Tensor, distance_map: torch.Tensor) -> torch.Tensor:
    """Kervadec-style regional integral of probabilities times SDF.

    ``probs`` and ``distance_map`` are (N, C, D, H, W). Background channel 0
    is ignored.
    """
    if probs.shape != distance_map.shape:
        raise ValueError("probs and distance_map must share shape")
    return (probs[:, 1:] * distance_map[:, 1:]).mean()


class CombinedSegLoss(nn.Module):
    def __init__(
        self,
        dice_weight: float = 1.0,
        ce_weight: float = 1.0,
        boundary_weight: float = 0.0,
        smooth: float = 1e-5,
        include_background: bool = False,
    ) -> None:
        super().__init__()
        self.dice_weight = dice_weight
        self.ce_weight = ce_weight
        self.boundary_weight = boundary_weight
        self.smooth = smooth
        self.include_background = include_background

    def forward(
        self,
        logits: torch.Tensor,
        target: torch.Tensor,
        distance_map: torch.Tensor | None = None,
    ) -> torch.Tensor:
        loss = self.dice_weight * dice_loss(
            logits,
            target,
            smooth=self.smooth,
            include_background=self.include_background,
        )
        if target.ndim == logits.ndim:
            ce_target = target.argmax(dim=1)
        else:
            ce_target = target.long()
        loss = loss + self.ce_weight * F.cross_entropy(logits, ce_target)
        if self.boundary_weight > 0:
            if distance_map is None:
                raise ValueError("boundary_weight > 0 requires a distance_map")
            probs = torch.softmax(logits, dim=1)
            loss = loss + self.boundary_weight * boundary_loss(probs, distance_map)
        return loss
