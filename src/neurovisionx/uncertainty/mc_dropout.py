"""Monte Carlo dropout uncertainty (Gal and Ghahramani, ICML 2016).

Patient-level aggregation (mean entropy) is the quantity Jungo et al. (2020)
found more useful than voxel-wise error localization. This is not a clinical
confidence score.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def _enable_dropout(module: nn.Module) -> None:
    if isinstance(module, (nn.Dropout, nn.Dropout3d, nn.Dropout2d)):
        module.train()


@torch.no_grad()
def mc_dropout_predict(
    model: nn.Module,
    volume: torch.Tensor,
    samples: int = 10,
) -> torch.Tensor:
    """Return mean softmax over ``samples`` stochastic forwards. Shape (C, D, H, W)."""
    was_training = model.training
    model.eval()
    model.apply(_enable_dropout)
    stacked = []
    for _ in range(samples):
        logits = model(volume)
        stacked.append(torch.softmax(logits, dim=1))
    mean_probs = torch.stack(stacked, dim=0).mean(dim=0)
    if not was_training:
        model.eval()
    else:
        model.train()
    return mean_probs[0]


def predictive_entropy(probs: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    return -(probs * (probs + eps).log()).sum(dim=0)


def patient_uncertainty_score(entropy_map: torch.Tensor, foreground: torch.Tensor) -> float:
    """Mean entropy on voxels predicted or labeled as foreground."""
    mask = foreground.bool()
    if not mask.any():
        return float(entropy_map.mean().item())
    return float(entropy_map[mask].mean().item())
