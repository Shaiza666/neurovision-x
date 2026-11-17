"""Optional multimodal fusion wrappers.

Early fusion (channel concatenation of T1, T1ce, T2, FLAIR) is performed by
the data loader and is the Phase 3 baseline. Squeeze-and-Excitation on those
four channels (Hu et al., CVPR 2018) is an optional stem in front of an
unchanged backbone. This is not a new fusion architecture.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class SqueezeExcite3D(nn.Module):
    """Channel squeeze-excitation on a 5D tensor (N, C, D, H, W).

    Reduction defaults to 2 because BraTS input width is 4 modalities; the
    original ImageNet SE block used reduction 16 on much wider feature maps.
    """

    def __init__(self, channels: int, reduction: int = 2) -> None:
        super().__init__()
        hidden = max(int(channels) // int(reduction), 1)
        self.fc1 = nn.Linear(channels, hidden)
        self.fc2 = nn.Linear(hidden, channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 5:
            raise ValueError("expected (N, C, D, H, W)")
        squeeze = x.mean(dim=(2, 3, 4))
        excite = torch.sigmoid(self.fc2(F.relu(self.fc1(squeeze))))
        return x * excite.view(x.shape[0], x.shape[1], 1, 1, 1)


class FusionStem(nn.Module):
    """SE recalibration of input modalities, then an unmodified backbone."""

    def __init__(
        self,
        backbone: nn.Module,
        in_channels: int = 4,
        reduction: int = 2,
    ) -> None:
        super().__init__()
        self.se = SqueezeExcite3D(in_channels, reduction=reduction)
        self.backbone = backbone

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(self.se(x))


def maybe_wrap_fusion(
    backbone: nn.Module,
    fusion: str | None,
    in_channels: int,
    reduction: int = 2,
) -> nn.Module:
    """Return ``backbone`` unchanged for early/concat fusion."""
    name = str(fusion or "early").lower().replace("-", "_")
    if name in {"early", "none", "concat", "concatenate"}:
        return backbone
    if name in {"se", "squeeze_excite", "se_stem"}:
        return FusionStem(backbone, in_channels=in_channels, reduction=reduction)
    raise ValueError(
        f"Unknown fusion {fusion!r}. Use 'early' (baseline concatenation) or 'se'."
    )
