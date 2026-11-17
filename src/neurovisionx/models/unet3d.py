"""3D U-Net (Çiçek et al., MICCAI 2016) implemented in PyTorch.

Optional instance norm and dropout follow common BraTS encoder–decoder
practice (Myronenko, 2019). This is an established architecture, not a
claimed novel backbone.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, dropout: float) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.InstanceNorm3d(out_ch),
            nn.PReLU(),
            nn.Dropout3d(dropout) if dropout > 0 else nn.Identity(),
            nn.Conv3d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.InstanceNorm3d(out_ch),
            nn.PReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class UNet3D(nn.Module):
    def __init__(
        self,
        in_channels: int = 4,
        out_channels: int = 4,
        features: tuple[int, ...] | list[int] = (16, 32, 64, 128),
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        feats = list(features)
        self.encoders = nn.ModuleList()
        self.pools = nn.ModuleList()
        prev = in_channels
        for width in feats:
            self.encoders.append(ConvBlock(prev, width, dropout))
            self.pools.append(nn.MaxPool3d(2))
            prev = width

        self.bottleneck = ConvBlock(feats[-1], feats[-1] * 2, dropout)

        self.upconvs = nn.ModuleList()
        self.decoders = nn.ModuleList()
        prev = feats[-1] * 2
        for width in reversed(feats):
            self.upconvs.append(nn.ConvTranspose3d(prev, width, 2, stride=2))
            self.decoders.append(ConvBlock(width * 2, width, dropout))
            prev = width

        self.head = nn.Conv3d(feats[0], out_channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skips: list[torch.Tensor] = []
        current = x
        for encoder, pool in zip(self.encoders, self.pools):
            current = encoder(current)
            skips.append(current)
            current = pool(current)
        current = self.bottleneck(current)
        for up, decoder, skip in zip(self.upconvs, self.decoders, reversed(skips)):
            current = up(current)
            if current.shape[-3:] != skip.shape[-3:]:
                current = torch.nn.functional.interpolate(
                    current, size=skip.shape[-3:], mode="trilinear", align_corners=False
                )
            current = decoder(torch.cat([skip, current], dim=1))
        return self.head(current)
