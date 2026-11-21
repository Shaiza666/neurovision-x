"""Test-time augmentation by spatial flips (established TTA, not a new method).

H2 allows MC dropout *or* TTA. This module averages softmax after flipping
the volume along depth / height / width and flipping the output back.

Not a clinical confidence score. Not a BraTS result by itself.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from neurovisionx.uncertainty.mc_dropout import predictive_entropy

# Spatial axes of a (N, C, D, H, W) tensor.
SPATIAL_DIMS = (2, 3, 4)

# Identity plus all non-empty combinations of D/H/W flips (8 views).
SPATIAL_FLIP_SETS: tuple[tuple[int, ...], ...] = (
    (),
    (2,),
    (3,),
    (4,),
    (2, 3),
    (2, 4),
    (3, 4),
    (2, 3, 4),
)


def apply_flips(tensor: torch.Tensor, axes: tuple[int, ...]) -> torch.Tensor:
    """Flip ``tensor`` on the given 5D spatial axes. Empty ``axes`` is identity."""
    if tensor.ndim != 5:
        raise ValueError("apply_flips expects (N, C, D, H, W)")
    if not axes:
        return tensor
    return torch.flip(tensor, dims=axes)


def tta_view_axes(max_views: int = 8) -> tuple[tuple[int, ...], ...]:
    """Return 1–8 flip combinations. ``max_views=1`` is identity only."""
    n = max(1, min(int(max_views), len(SPATIAL_FLIP_SETS)))
    return SPATIAL_FLIP_SETS[:n]


@torch.no_grad()
def tta_stack(
    model: nn.Module,
    volume: torch.Tensor,
    max_views: int = 8,
    forward=None,
) -> torch.Tensor:
    """Softmax views of shape (T, C, D, H, W), already unflipped.

    ``forward`` if set maps a (1, C, D, H, W) volume to logits of the same
    spatial size (used for sliding-window TTA). Default is ``model``.
    """
    if volume.ndim == 4:
        volume = volume.unsqueeze(0)
    if volume.ndim != 5 or volume.shape[0] != 1:
        raise ValueError("volume must be (1, C, D, H, W) or (C, D, H, W)")
    run = forward if forward is not None else model
    was_training = model.training
    model.eval()
    stacked = []
    for axes in tta_view_axes(max_views):
        flipped = apply_flips(volume, axes)
        logits = run(flipped)
        if logits.ndim != 5:
            raise ValueError("TTA forward must return (N, C, D, H, W) logits")
        restored = apply_flips(logits, axes)
        stacked.append(torch.softmax(restored, dim=1)[0])
    if was_training:
        model.train()
    return torch.stack(stacked, dim=0)


@torch.no_grad()
def tta_stats(
    model: nn.Module,
    volume: torch.Tensor,
    max_views: int = 8,
    forward=None,
) -> dict[str, torch.Tensor]:
    """Mean softmax, entropy, mutual information, and class-mean variance."""
    stack = tta_stack(model, volume, max_views=max_views, forward=forward)
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
def tta_predict(
    model: nn.Module,
    volume: torch.Tensor,
    max_views: int = 8,
    forward=None,
) -> torch.Tensor:
    """Mean softmax over TTA views. Shape (C, D, H, W)."""
    return tta_stats(model, volume, max_views=max_views, forward=forward)["mean_probs"]


def normalize_uncertainty_method(method: str | None) -> str:
    name = str(method or "mc_dropout").lower().replace("-", "_")
    if name in {"mc", "mc_dropout", "dropout"}:
        return "mc_dropout"
    if name in {"tta", "test_time_augmentation", "flip_tta"}:
        return "tta"
    if name in {"none", "off", "deterministic"}:
        return "none"
    raise ValueError(
        f"Unknown training.uncertainty.method {method!r}. "
        "Use mc_dropout, tta, or none."
    )
