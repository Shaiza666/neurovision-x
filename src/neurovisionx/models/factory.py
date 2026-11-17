"""Build segmentation models from config.

MONAI UNet is used when ``optional_monai`` is true and MONAI is installed.
Otherwise the in-repo 3D U-Net is used. Failures raise; they are not replaced
with dummy logits.
"""

from __future__ import annotations

from typing import Any

import torch.nn as nn

from neurovisionx.models.unet3d import UNet3D


def build_model(model_cfg: dict[str, Any]) -> nn.Module:
    name = str(model_cfg.get("name", "unet3d")).lower()
    in_ch = int(model_cfg.get("in_channels", 4))
    out_ch = int(model_cfg.get("out_channels", 4))
    features = model_cfg.get("features", [16, 32, 64, 128])
    dropout = float(model_cfg.get("dropout", 0.1))

    if name in {"unetr", "swinunetr"}:
        raise NotImplementedError(
            f"{name} is optional (hypothesis H3) and is not enabled until a "
            "matched GPU comparison is approved."
        )

    if model_cfg.get("optional_monai") and name == "unet3d":
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

    return UNet3D(
        in_channels=in_ch,
        out_channels=out_ch,
        features=tuple(features),
        dropout=dropout,
    )
