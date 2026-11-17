"""Validation engine: loss plus nested BraTS Dice (and optional HD95).

Patch-wise scores during training are optimization signals, not challenge
leaderboard numbers. Full-volume Dice/HD95 belong in ``scripts/evaluate.py``
after a real checkpoint exists.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
import torch.nn as nn

from neurovisionx.evaluation.regions import region_scores


def _maybe_distance(criterion: nn.Module, logits: torch.Tensor, labels: torch.Tensor):
    if getattr(criterion, "boundary_weight", 0) <= 0:
        return None
    from neurovisionx.data.distance import one_hot_distance_map

    return one_hot_distance_map(labels, logits.shape[1])


def scores_from_logits(
    logits: torch.Tensor,
    labels: torch.Tensor,
    *,
    compute_hd95: bool = False,
) -> dict[str, float]:
    pred = logits.argmax(dim=1).detach().cpu().numpy()
    target = labels.detach().cpu().numpy()
    if target.ndim == logits.ndim:
        target = target.argmax(axis=1)
    acc: dict[str, list[float]] = {}
    for index in range(pred.shape[0]):
        row = region_scores(pred[index], target[index], compute_hd95=compute_hd95)
        for key, value in row.items():
            acc.setdefault(key, []).append(float(value))
    return {key: float(np.nanmean(values)) for key, values in acc.items()}


@torch.no_grad()
def validate_epoch(
    model: nn.Module,
    loader: Any,
    criterion: nn.Module,
    device: torch.device,
    *,
    compute_hd95: bool = False,
) -> dict[str, float]:
    model.eval()
    losses: list[float] = []
    metric_lists: dict[str, list[float]] = {}
    for batch in loader:
        images = batch["image"].to(device)
        labels = batch["label"].to(device)
        logits = model(images)
        loss = criterion(logits, labels, distance_map=_maybe_distance(criterion, logits, labels))
        losses.append(float(loss.item()))
        row = scores_from_logits(logits, labels, compute_hd95=compute_hd95)
        for key, value in row.items():
            metric_lists.setdefault(key, []).append(value)
    out = {"val_loss": float(sum(losses) / max(len(losses), 1))}
    for key, values in metric_lists.items():
        out[key] = float(np.nanmean(values))
    return out
