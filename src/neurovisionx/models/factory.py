"""Build segmentation models from config.

MONAI UNet / AttentionUnet are used when ``optional_monai`` is true and MONAI
is installed. Otherwise the in-repo 3D U-Net or Attention U-Net is used.
Failures raise; they are not replaced with dummy logits.
"""

from __future__ import annotations

from typing import Any

import torch.nn as nn

from neurovisionx.models.attention_unet3d import AttentionUNet3D
from neurovisionx.models.fusion import maybe_wrap_fusion
from neurovisionx.models.unet3d import UNet3D


def _backbone(model_cfg: dict[str, Any]) -> nn.Module:
    name = str(model_cfg.get("name", "unet3d")).lower().replace("-", "_")
    in_ch = int(model_cfg.get("in_channels", 4))
    out_ch = int(model_cfg.get("out_channels", 4))
    features = model_cfg.get("features", [16, 32, 64, 128])
    dropout = float(model_cfg.get("dropout", 0.1))
    use_monai = bool(model_cfg.get("optional_monai"))

    if name in {"unetr", "swinunetr"}:
        raise NotImplementedError(
            f"{name} is optional (hypothesis H3) and is not enabled until a "
            "matched GPU comparison is approved."
        )

    if name in {"attention_unet3d", "attentionunet", "attention_unet"}:
        if use_monai:
            try:
                from monai.networks.nets import AttentionUnet as MonaiAttentionUnet

                return MonaiAttentionUnet(
                    spatial_dims=3,
                    in_channels=in_ch,
                    out_channels=out_ch,
                    channels=tuple(features),
                    strides=tuple([2] * (len(features) - 1)),
                    dropout=dropout,
                )
            except ImportError:
                pass
        return AttentionUNet3D(
            in_channels=in_ch,
            out_channels=out_ch,
            features=tuple(features),
            dropout=dropout,
        )

    if use_monai and name == "unet3d":
        try:
            from monai.networks.nets import UNet as MonaiUNet

            return MonaiUNet(
                spatial_dims=3,
                in_channels=in_ch,
                out_channels=out_ch,
                channels=tuple(features),
                strides=tuple([2] * (len(features) - 1)),
                num_res_units=int(model_cfg.get("num_res_units", 2)),
                dropout=dropout,
            )
        except ImportError:
            pass

    if name not in {"unet3d", "unet", "3dunet"}:
        raise ValueError(f"Unknown model name {name!r}")

    return UNet3D(
        in_channels=in_ch,
        out_channels=out_ch,
        features=tuple(features),
        dropout=dropout,
    )


def build_model(model_cfg: dict[str, Any]) -> nn.Module:
    """Build the configured backbone, then optionally wrap an SE fusion stem.

    ``fusion: early`` (default) returns the Phase 3 model unchanged.
    """
    backbone = _backbone(model_cfg)
    return maybe_wrap_fusion(
        backbone,
        model_cfg.get("fusion", "early"),
        in_channels=int(model_cfg.get("in_channels", 4)),
        reduction=int(model_cfg.get("fusion_reduction", 2)),
    )
