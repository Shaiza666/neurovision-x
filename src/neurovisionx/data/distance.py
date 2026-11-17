"""Signed distance maps for Kervadec-style boundary loss.

Distance maps are derived from real (or test-synthetic tensor) labels. They
are not a substitute for missing BraTS volumes.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F


def class_signed_distance(mask: np.ndarray) -> np.ndarray:
    """SDF for one binary mask: negative inside, positive outside."""
    try:
        from scipy.ndimage import distance_transform_edt
    except ImportError as exc:
        raise ImportError("scipy is required for boundary distance maps") from exc
    inside = np.asarray(mask, dtype=bool)
    if inside.all() or not inside.any():
        return np.zeros(inside.shape, dtype=np.float32)
    outside = np.logical_not(inside)
    pos = distance_transform_edt(outside)
    neg = distance_transform_edt(inside)
    return (pos - neg).astype(np.float32)


def one_hot_distance_map(target: torch.Tensor, n_classes: int) -> torch.Tensor:
    """Build (N, C, D, H, W) SDFs from integer labels (N, D, H, W)."""
    if target.ndim == 5:
        one_hot = target
    else:
        one_hot = F.one_hot(target.long(), n_classes).permute(0, 4, 1, 2, 3).float()
    maps = []
    for batch in one_hot:
        channels = [class_signed_distance(channel.detach().cpu().numpy() > 0.5) for channel in batch]
        maps.append(torch.from_numpy(np.stack(channels, axis=0)))
    stacked = torch.stack(maps, dim=0).to(device=target.device, dtype=torch.float32)
    return stacked
