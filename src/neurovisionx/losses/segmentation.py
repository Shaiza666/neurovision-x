"""Segmentation losses.

Dice (Milletari et al., 2016) is a regional overlap objective:

    Dice_c = 2 <p, g> / (|p|_1 + |g|_1)

Class-wise Dice (background excluded by default) already down-weights the
dominant background class. Classes with no ground-truth voxels in the batch are
omitted from the Dice mean so softmax leakage on absent labels does not
dominate. Cross-entropy can additionally use inverse-frequency weights
(Sudre et al., 2017 discuss related GDL).

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
    pred_c = probs[:, start:]
    target_c = one_hot[:, start:]
    intersection = (pred_c * target_c).sum(dim=dims)
    union = pred_c.sum(dim=dims) + target_c.sum(dim=dims)
    dice = (2 * intersection + smooth) / (union + smooth)
    present = target_c.sum(dim=dims) > 0
    if not bool(present.any()):
        return logits.sum() * 0.0
    return 1.0 - dice[present].mean()


def boundary_loss(
    probs: torch.Tensor,
    distance_map: torch.Tensor,
    classes: list[int] | tuple[int, ...] | None = None,
) -> torch.Tensor:
    """Kervadec-style regional integral of probabilities times SDF.

    ``probs`` and ``distance_map`` are (N, C, D, H, W). Background channel 0
    is ignored unless it is the only requested class. ``classes=None`` uses
    every foreground channel (Phase 3 behaviour). Pass ``[3]`` to restrict
    the term to ET for hypothesis H1.
    """
    if probs.shape != distance_map.shape:
        raise ValueError("probs and distance_map must share shape")
    if classes is None:
        channels = list(range(1, probs.shape[1]))
    else:
        channels = [int(c) for c in classes if int(c) != 0]
    if not channels:
        return probs.sum() * 0.0
    return (probs[:, channels] * distance_map[:, channels]).mean()


def inverse_frequency_weights(target: torch.Tensor, n_classes: int) -> torch.Tensor:
    """Per-batch inverse-frequency CE weights. Counts of 0 are treated as 1."""
    if target.ndim == 5:
        labels = target.argmax(dim=1)
    else:
        labels = target.long()
    counts = torch.stack([(labels == class_id).sum() for class_id in range(n_classes)]).to(
        dtype=torch.float32, device=target.device
    )
    counts = torch.clamp(counts, min=1.0)
    return counts.sum() / (float(n_classes) * counts)


class CombinedSegLoss(nn.Module):
    def __init__(
        self,
        dice_weight: float = 1.0,
        ce_weight: float = 1.0,
        boundary_weight: float = 0.0,
        smooth: float = 1e-5,
        include_background: bool = False,
        class_weight_mode: str = "none",
        class_weights: list[float] | tuple[float, ...] | None = None,
        boundary_classes: list[int] | tuple[int, ...] | None = None,
    ) -> None:
        super().__init__()
        self.dice_weight = dice_weight
        self.ce_weight = ce_weight
        self.boundary_weight = boundary_weight
        self.smooth = smooth
        self.include_background = include_background
        self.class_weight_mode = str(class_weight_mode).lower()
        self.boundary_classes = None if boundary_classes is None else tuple(int(c) for c in boundary_classes)
        if class_weights is not None:
            self.register_buffer("fixed_class_weights", torch.tensor(class_weights, dtype=torch.float32))
        else:
            self.fixed_class_weights = None

    def _ce_weight(self, target: torch.Tensor, n_classes: int) -> torch.Tensor | None:
        if self.class_weight_mode in {"inverse_freq", "inverse_frequency"}:
            return inverse_frequency_weights(target, n_classes)
        if self.class_weight_mode in {"fixed", "provided"} and self.fixed_class_weights is not None:
            return self.fixed_class_weights.to(device=target.device, dtype=torch.float32)
        if self.fixed_class_weights is not None and self.class_weight_mode == "none":
            return self.fixed_class_weights.to(device=target.device, dtype=torch.float32)
        return None

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
        ce_w = self._ce_weight(ce_target, logits.shape[1])
        loss = loss + self.ce_weight * F.cross_entropy(logits, ce_target, weight=ce_w)
        if self.boundary_weight > 0:
            if distance_map is None:
                raise ValueError("boundary_weight > 0 requires a distance_map")
            probs = torch.softmax(logits, dim=1)
            loss = loss + self.boundary_weight * boundary_loss(
                probs, distance_map, classes=self.boundary_classes
            )
        return loss
