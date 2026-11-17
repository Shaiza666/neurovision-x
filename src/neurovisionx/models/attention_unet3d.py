"""3D Attention U-Net (Oktay et al., MIDL 2018) for volumetric MRI.

Additive attention gates on skip connections suppress encoder features that
are poorly aligned with the decoder. This is an established architecture,
not a claimed NeuroVision-X backbone. Original paper is 2D CT; the 3D
extension here is the standard channel-wise gate used in MONAI AttentionUnet.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from neurovisionx.models.unet3d import ConvBlock


class GridAttentionGate3D(nn.Module):
    """Additive attention gate (Oktay et al., 2018)."""

    def __init__(self, in_channels: int, gate_channels: int, inter_channels: int | None = None) -> None:
        super().__init__()
        inter = inter_channels or max(in_channels // 2, 1)
        self.W_x = nn.Conv3d(in_channels, inter, kernel_size=1, bias=False)
        self.W_g = nn.Conv3d(gate_channels, inter, kernel_size=1, bias=True)
        self.psi = nn.Conv3d(inter, 1, kernel_size=1, bias=True)

    def forward(self, skip: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
        if gate.shape[-3:] != skip.shape[-3:]:
            gate = F.interpolate(gate, size=skip.shape[-3:], mode="trilinear", align_corners=False)
        attention = F.relu(self.W_x(skip) + self.W_g(gate), inplace=True)
        alpha = torch.sigmoid(self.psi(attention))
        return skip * alpha


class AttentionUNet3D(nn.Module):
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
        self.gates = nn.ModuleList()
        self.decoders = nn.ModuleList()
        prev = feats[-1] * 2
        for width in reversed(feats):
            self.upconvs.append(nn.ConvTranspose3d(prev, width, 2, stride=2))
            self.gates.append(GridAttentionGate3D(in_channels=width, gate_channels=width))
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
        for up, gate, decoder, skip in zip(self.upconvs, self.gates, self.decoders, reversed(skips)):
            current = up(current)
            if current.shape[-3:] != skip.shape[-3:]:
                current = F.interpolate(current, size=skip.shape[-3:], mode="trilinear", align_corners=False)
            attended = gate(skip, current)
            current = decoder(torch.cat([attended, current], dim=1))
        return self.head(current)
