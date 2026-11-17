"""3D Grad-CAM (Selvaraju et al., ICCV 2017) for volumetric logits.

The map is a research localization of gradient-weighted activations. It is
not a causal explanation, not a proof of correctness, and not a clinical
rationale. See docs/explainability.md.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F


DISCLAIMER = (
    "Research visualization. Not a clinical explanation. "
    "Grad-CAM does not prove model reasoning or correctness."
)


@dataclass
class GradCAMResult:
    """Attribution tied to the same forward pass as ``prediction``."""

    prediction: torch.Tensor  # (D, H, W) int
    logits: torch.Tensor  # (C, D, H, W)
    cam: torch.Tensor  # (D, H, W) in [0, 1]
    class_index: int
    layer_name: str
    score: float
    score_mode: str
    disclaimer: str = field(default=DISCLAIMER)


def named_conv3d(model: nn.Module) -> list[tuple[str, nn.Conv3d]]:
    return [(name, module) for name, module in model.named_modules() if isinstance(module, nn.Conv3d)]


def select_cam_layer(model: nn.Module, layer_name: str | None = None) -> tuple[str, nn.Conv3d]:
    """Prefer the last spatial (k>1) Conv3d; skip a 1x1 head when possible."""
    convs = named_conv3d(model)
    if not convs:
        raise ValueError("No Conv3d layer found for Grad-CAM.")
    if layer_name:
        lookup = dict(convs)
        if layer_name not in lookup:
            raise KeyError(f"Unknown CAM layer {layer_name!r}. Known: {list(lookup)}")
        return layer_name, lookup[layer_name]
    spatial = [(name, module) for name, module in convs if int(module.kernel_size[0]) > 1]
    if spatial:
        return spatial[-1]
    return convs[-1]


def gradcam_3d(
    model: nn.Module,
    volume: torch.Tensor,
    *,
    class_index: int = 3,
    layer_name: str | None = None,
) -> GradCAMResult:
    """CAM for class ``class_index`` from one differentiable forward.

    Score is the mean logit on voxels predicted as that class (links CAM to
    the actual segmentation). If the class is empty, the mean logit over the
    whole volume is used and ``score_mode`` records the fallback.
    """
    if volume.ndim == 4:
        volume = volume.unsqueeze(0)
    if volume.ndim != 5:
        raise ValueError("volume must be (1, C, D, H, W) or (C, D, H, W)")

    name, layer = select_cam_layer(model, layer_name)
    stored: dict[str, torch.Tensor] = {}

    def _forward_hook(_module: nn.Module, _inputs: tuple, output: torch.Tensor) -> None:
        stored["activation"] = output

    def _backward_hook(_module: nn.Module, _grad_input: tuple, grad_output: tuple) -> None:
        stored["gradient"] = grad_output[0]

    handle_f = layer.register_forward_hook(_forward_hook)
    handle_b = layer.register_full_backward_hook(_backward_hook)
    was_training = model.training
    model.eval()
    try:
        logits = model(volume)
        prediction = logits.argmax(dim=1)
        class_map = logits[:, class_index]
        region = (prediction == class_index).to(dtype=class_map.dtype)
        if float(region.sum().item()) > 0:
            score = (class_map * region).sum() / region.sum()
            score_mode = "predicted_region_mean"
        else:
            score = class_map.mean()
            score_mode = "class_logit_mean_empty_pred"
        model.zero_grad(set_to_none=True)
        if volume.requires_grad:
            if volume.grad is not None:
                volume.grad.zero_()
        score.backward()
        if "activation" not in stored or "gradient" not in stored:
            raise RuntimeError("Grad-CAM hooks did not fire; choose another Conv3d layer.")
        activation = stored["activation"]
        gradient = stored["gradient"]
        weights = gradient.mean(dim=(2, 3, 4), keepdim=True)
        cam = torch.relu((weights * activation).sum(dim=1, keepdim=True))
        cam = F.interpolate(cam, size=volume.shape[-3:], mode="trilinear", align_corners=False)
        cam = cam[0, 0]
        peak = cam.max().clamp(min=1e-8)
        cam = cam / peak
    finally:
        handle_f.remove()
        handle_b.remove()
        if was_training:
            model.train()
        else:
            model.eval()
        model.zero_grad(set_to_none=True)

    return GradCAMResult(
        prediction=prediction[0].detach(),
        logits=logits[0].detach(),
        cam=cam.detach(),
        class_index=int(class_index),
        layer_name=name,
        score=float(score.detach().item()),
        score_mode=score_mode,
    )


def input_gradient_saliency(
    model: nn.Module,
    volume: torch.Tensor,
    *,
    class_index: int = 3,
) -> torch.Tensor:
    """Mean absolute input gradient over channels. Not a clinical explanation."""
    if volume.ndim == 4:
        volume = volume.unsqueeze(0)
    volume = volume.detach().clone().requires_grad_(True)
    was_training = model.training
    model.eval()
    try:
        logits = model(volume)
        prediction = logits.argmax(dim=1)
        class_map = logits[:, class_index]
        region = (prediction == class_index).to(dtype=class_map.dtype)
        score = (class_map * region).sum() / region.sum().clamp(min=1.0)
        model.zero_grad(set_to_none=True)
        score.backward()
        if volume.grad is None:
            raise RuntimeError("Input gradient is None.")
        saliency = volume.grad.abs().mean(dim=1)[0]
        peak = saliency.max().clamp(min=1e-8)
        return (saliency / peak).detach()
    finally:
        if was_training:
            model.train()
        else:
            model.eval()
        model.zero_grad(set_to_none=True)
