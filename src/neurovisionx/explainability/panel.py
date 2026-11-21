"""Build a full explainability panel from one model pass.

Predictions, Grad-CAM, and optional entropy are stored together so figures
cannot drift from a different forward.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn

from neurovisionx.explainability.gradcam import DISCLAIMER, GradCAMResult, gradcam_3d, input_gradient_saliency
from neurovisionx.explainability.occlusion import occlusion_sensitivity
from neurovisionx.explainability.overlays import save_orthogonal_attribution, save_orthogonal_overlay
from neurovisionx.explainability.volume3d import save_prediction_pointcloud


def softmax_entropy(logits: np.ndarray) -> np.ndarray:
    """T=1 predictive entropy from the Grad-CAM logits (nats). Not MC dropout."""
    shifted = logits - np.max(logits, axis=0, keepdims=True)
    exp = np.exp(shifted)
    probs = exp / np.clip(exp.sum(axis=0, keepdims=True), 1e-8, None)
    return -(probs * np.log(np.clip(probs, 1e-8, 1.0))).sum(axis=0)


def explanation_to_numpy(result: GradCAMResult) -> dict[str, Any]:
    return {
        "prediction": result.prediction.detach().cpu().numpy().astype(np.int16),
        "cam": result.cam.detach().cpu().numpy().astype(np.float32),
        "class_index": result.class_index,
        "layer_name": result.layer_name,
        "score": result.score,
        "score_mode": result.score_mode,
        "disclaimer": result.disclaimer,
    }


def write_explain_panel(
    image: np.ndarray,
    result: GradCAMResult,
    out_dir: Path | str,
    *,
    target: np.ndarray | None = None,
    entropy: np.ndarray | None = None,
    saliency: np.ndarray | None = None,
    occlusion: np.ndarray | None = None,
    title: str = DISCLAIMER,
) -> dict[str, str]:
    """Write overlay, attribution, 3D preview, and a JSON sidecar."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    payload = explanation_to_numpy(result)
    pred = payload["prediction"]
    cam = payload["cam"]
    if entropy is None:
        entropy = softmax_entropy(result.logits.detach().cpu().numpy())
        entropy_kind = "softmax_entropy_T1_same_forward"
    else:
        entropy_kind = "provided_map"
    paths = {
        "overlay": str(
            save_orthogonal_overlay(image, pred, out / "orthogonal_overlay.png", target=target, title=title)
        ),
        "attribution": str(
            save_orthogonal_attribution(
                image,
                cam,
                out / "orthogonal_attribution.png",
                entropy=entropy,
                title=title,
            )
        ),
        "volume3d": str(save_prediction_pointcloud(pred, out / "prediction_3d_preview.png", title=title)),
    }
    if saliency is not None:
        paths["gradient"] = str(
            save_orthogonal_attribution(
                image,
                saliency,
                out / "orthogonal_input_gradient.png",
                cam_label="input |grad| (normalised, unitless)",
                title=title,
            )
        )
    if occlusion is not None:
        paths["occlusion"] = str(
            save_orthogonal_attribution(
                image,
                occlusion,
                out / "orthogonal_occlusion.png",
                cam_label="occlusion sensitivity (normalised, unitless)",
                title=title,
            )
        )
    sidecar = {
        "class_index": payload["class_index"],
        "layer_name": payload["layer_name"],
        "score": payload["score"],
        "score_mode": payload["score_mode"],
        "has_ground_truth": target is not None,
        "has_entropy": True,
        "entropy_kind": entropy_kind,
        "has_occlusion": occlusion is not None,
        "files": paths,
        "disclaimer": DISCLAIMER,
        "limitations_doc": "docs/explainability.md",
    }
    (out / "explain.json").write_text(json.dumps(sidecar, indent=2), encoding="utf-8")
    paths["sidecar"] = str(out / "explain.json")
    return paths


def explain_volume(
    model: nn.Module,
    volume: torch.Tensor,
    image: np.ndarray,
    out_dir: Path | str,
    *,
    target: np.ndarray | None = None,
    class_index: int = 3,
    entropy: np.ndarray | None = None,
    also_gradient: bool = False,
    also_occlusion: bool = False,
    title: str = DISCLAIMER,
) -> dict[str, str]:
    """Grad-CAM on ``volume``, figures from the same prediction."""
    result = gradcam_3d(model, volume, class_index=class_index)
    saliency = None
    if also_gradient:
        saliency = input_gradient_saliency(model, volume, class_index=class_index).cpu().numpy()
    occlusion = None
    if also_occlusion:
        occ = occlusion_sensitivity(model, volume, patch=8, stride=8, class_index=class_index)
        peak = occ.max().clamp(min=1e-8)
        occlusion = (occ / peak).detach().cpu().numpy()
    return write_explain_panel(
        image,
        result,
        out_dir,
        target=target,
        entropy=entropy,
        saliency=saliency,
        occlusion=occlusion,
        title=title,
    )
