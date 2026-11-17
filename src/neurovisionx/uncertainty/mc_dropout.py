"""Monte Carlo dropout uncertainty (Gal and Ghahramani, ICML 2016).

Patient-level aggregation (mean entropy) is the quantity Jungo et al. (2020)
found more useful than voxel-wise error localization. This is not a clinical
confidence score.
"""

from __future__ import annotations

import torch
import torch.nn as nn


def _enable_dropout(module: nn.Module) -> None:
    if isinstance(module, (nn.Dropout, nn.Dropout3d, nn.Dropout2d)):
        module.train()


def enable_mc_dropout(model: nn.Module) -> nn.Module:
    """Eval mode with dropout left on (Gal and Ghahramani, 2016)."""
    model.eval()
    model.apply(_enable_dropout)
    return model


@torch.no_grad()
def mc_dropout_stack(
    model: nn.Module,
    volume: torch.Tensor,
    samples: int = 10,
) -> torch.Tensor:
    """Softmax samples of shape (T, C, D, H, W)."""
    if volume.ndim == 4:
        volume = volume.unsqueeze(0)
    was_training = model.training
    enable_mc_dropout(model)
    stacked = []
    for _ in range(max(int(samples), 1)):
        logits = model(volume)
        stacked.append(torch.softmax(logits, dim=1)[0])
    if was_training:
        model.train()
    else:
        model.eval()
    return torch.stack(stacked, dim=0)


@torch.no_grad()
def mc_dropout_stats(
    model: nn.Module,
    volume: torch.Tensor,
    samples: int = 10,
) -> dict[str, torch.Tensor]:
    """Mean softmax, entropy, mutual information, and class-mean variance."""
    stack = mc_dropout_stack(model, volume, samples=samples)
    mean_probs = stack.mean(dim=0)
    entropy = predictive_entropy(mean_probs)
    per_sample = torch.stack([predictive_entropy(sample) for sample in stack], dim=0)
    mutual_info = torch.clamp(entropy - per_sample.mean(dim=0), min=0.0)
    variance = stack.var(dim=0, unbiased=False).mean(dim=0)
    return {
        "mean_probs": mean_probs,
        "entropy": entropy,
        "mutual_info": mutual_info,
        "variance": variance,
        "samples": stack,
    }


@torch.no_grad()
def mc_dropout_predict(
    model: nn.Module,
    volume: torch.Tensor,
    samples: int = 10,
) -> torch.Tensor:
    """Return mean softmax over ``samples`` stochastic forwards. Shape (C, D, H, W)."""
    return mc_dropout_stats(model, volume, samples=samples)["mean_probs"]


def predictive_entropy(probs: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    return -(probs * (probs + eps).log()).sum(dim=0)


def patient_uncertainty_score(entropy_map: torch.Tensor, foreground: torch.Tensor) -> float:
    """Mean entropy on voxels predicted or labeled as foreground."""
    mask = foreground.bool()
    if not mask.any():
        return float(entropy_map.mean().item())
    return float(entropy_map[mask].mean().item())
