"""Actual inference + research visualizations for the demo UI.

Never substitutes random logits for a failed load. Smoke checkpoints must be
labelled synthetic in the returned payload.
"""

from __future__ import annotations

import io
import json
import time
import zipfile
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn

from neurovisionx.demo.checkpoints import patch_size_from_payload
from neurovisionx.evaluation.cost import count_parameters
from neurovisionx.evaluation.inference import sliding_window_logits
from neurovisionx.evaluation.regions import region_scores
from neurovisionx.explainability.gradcam import DISCLAIMER, gradcam_3d
from neurovisionx.explainability.occlusion import occlusion_sensitivity
from neurovisionx.explainability.overlays import (
    error_rgb,
    labels_to_rgb,
    mri_plane,
)
from neurovisionx.explainability.panel import softmax_entropy, write_explain_panel
from neurovisionx.explainability.views import ORIENTATION_AXIS, ORIENTATION_ORDER, extract_slice
from neurovisionx.uncertainty.mc_dropout import mc_dropout_stats

CAM_VOXEL_LIMIT = 48 ** 3
OCCLUSION_VOXEL_LIMIT = 24 ** 3
MC_VOXEL_LIMIT = 48 ** 3


def _normalize_map(tensor: torch.Tensor) -> np.ndarray:
    array = tensor.detach().cpu().float()
    peak = array.max().clamp(min=1e-8)
    return (array / peak).numpy()


def _to_nchw(image: np.ndarray) -> torch.Tensor:
    array = np.ascontiguousarray(image, dtype=np.float32)
    tensor = torch.from_numpy(array)
    if tensor.ndim == 4:
        tensor = tensor.unsqueeze(0)
    return tensor


def forward_logits(
    model: nn.Module,
    volume: torch.Tensor,
    patch: tuple[int, int, int],
    overlap: float = 0.5,
) -> torch.Tensor:
    """Logits of the loaded model. Sliding window if the volume exceeds the patch."""
    if volume.ndim == 4:
        volume = volume.unsqueeze(0)
    spatial = tuple(int(v) for v in volume.shape[-3:])
    if spatial == patch:
        model.eval()
        with torch.no_grad():
            return model(volume)
    return sliding_window_logits(model, volume, patch, overlap=overlap)


def run_inference(
    model: nn.Module,
    payload: dict[str, Any],
    image: np.ndarray,
    *,
    target: np.ndarray | None = None,
    class_index: int = 3,
    overlap: float = 0.5,
    compute_cam: bool = True,
    compute_occlusion: bool = False,
    compute_mc: bool = False,
    mc_samples: int = 5,
    device: torch.device | None = None,
) -> dict[str, Any]:
    """Run the checkpoint on ``image`` (C,D,H,W). Returns arrays from that pass.

    Displayed labels stay the T=1 argmax. Optional MC entropy and occlusion
    maps are extra research views; they do not replace the segmentation.
    """
    device = device or torch.device("cpu")
    model = model.to(device)
    model.eval()
    patch = patch_size_from_payload(payload)
    volume = _to_nchw(image).to(device)
    spatial = tuple(int(v) for v in volume.shape[-3:])
    n_voxels = int(np.prod(spatial))
    t0 = time.perf_counter()
    logits = forward_logits(model, volume, patch, overlap=overlap)
    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    pred = logits.argmax(dim=1)[0].detach().cpu().numpy().astype(np.int16)
    logits_np = logits[0].detach().cpu().numpy()
    entropy = softmax_entropy(logits_np)
    cam = None
    cam_note = "not computed"
    if compute_cam and n_voxels <= CAM_VOXEL_LIMIT:
        try:
            cam_result = gradcam_3d(model, volume, class_index=class_index)
            cam = cam_result.cam.detach().cpu().numpy()
            cam_note = (
                f"3D Grad-CAM on full volume, layer={cam_result.layer_name}, "
                f"score_mode={cam_result.score_mode}"
            )
        except Exception as exc:
            cam = None
            cam_note = f"Grad-CAM failed ({type(exc).__name__}): {exc}"
        # Keep prediction from the timed no-grad pass so CAM backward cannot
        # silently replace the displayed segmentation.
    elif compute_cam:
        cam_note = (
            f"Grad-CAM skipped: volume has {n_voxels} voxels "
            f"(limit {CAM_VOXEL_LIMIT}). Attribution needs a smaller crop."
        )

    occlusion = None
    occlusion_note = "not computed"
    if compute_occlusion and n_voxels <= OCCLUSION_VOXEL_LIMIT:
        try:
            occ = occlusion_sensitivity(
                model, volume, patch=8, stride=8, class_index=class_index
            )
            occlusion = _normalize_map(occ)
            occlusion_note = (
                f"Occlusion sensitivity (8³ cubes, stride 8) for class {class_index}. "
                "Zero cubes are not MRI; this is not a clinical explanation."
            )
        except Exception as exc:
            occlusion = None
            occlusion_note = f"Occlusion failed ({type(exc).__name__}): {exc}"
    elif compute_occlusion:
        occlusion_note = (
            f"Occlusion skipped: volume has {n_voxels} voxels "
            f"(limit {OCCLUSION_VOXEL_LIMIT}). Use a crop or scripts/explain.py."
        )

    mc_entropy = None
    mc_note = "not computed"
    if compute_mc and n_voxels <= MC_VOXEL_LIMIT and spatial == patch:
        try:
            stats = mc_dropout_stats(model, volume, samples=max(int(mc_samples), 1))
            mc_entropy = stats["entropy"].detach().cpu().numpy()
            mc_note = (
                f"MC dropout T={max(int(mc_samples), 1)} predictive entropy. "
                "Displayed labels remain T=1 argmax. Not a clinical confidence score."
            )
        except Exception as exc:
            mc_entropy = None
            mc_note = f"MC dropout failed ({type(exc).__name__}): {exc}"
        finally:
            model.eval()
    elif compute_mc and spatial != patch:
        mc_note = (
            "MC dropout skipped for sliding-window volumes. "
            "Use scripts/analyze_reliability.py after licensed BraTS data is available."
        )
    elif compute_mc:
        mc_note = (
            f"MC dropout skipped: volume has {n_voxels} voxels "
            f"(limit {MC_VOXEL_LIMIT})."
        )

    scores = None
    if target is not None:
        try:
            scores = region_scores(pred, target, compute_hd95=True, compute_pr=True)
        except ImportError:
            scores = region_scores(pred, target, compute_hd95=False, compute_pr=True)
            scores["hd95_note"] = "scipy not installed; HD95 omitted"

    unique, counts = np.unique(pred, return_counts=True)
    class_hist = {int(k): int(v) for k, v in zip(unique, counts)}
    smoke = bool(payload.get("smoke"))
    return {
        "prediction": pred,
        "logits": logits_np,
        "entropy": entropy,
        "cam": cam,
        "cam_note": cam_note,
        "occlusion": occlusion,
        "occlusion_note": occlusion_note,
        "mc_entropy": mc_entropy,
        "mc_note": mc_note,
        "elapsed_ms": float(elapsed_ms),
        "n_params": count_parameters(model),
        "patch": list(patch),
        "device": str(device),
        "smoke": smoke,
        "class_index": int(class_index),
        "class_hist": class_hist,
        "scores": scores,
        "disclaimer": DISCLAIMER,
        "status_banner": (
            "SMOKE / SYNTHETIC CHECKPOINT — not BraTS, not a clinical result."
            if smoke
            else "Research prototype. Not a clinically validated diagnostic system."
        ),
    }


def slice_pack(
    image: np.ndarray,
    prediction: np.ndarray,
    *,
    target: np.ndarray | None = None,
    entropy: np.ndarray | None = None,
    cam: np.ndarray | None = None,
    occlusion: np.ndarray | None = None,
    mc_entropy: np.ndarray | None = None,
    indices: dict[str, int] | None = None,
) -> dict[str, dict[str, np.ndarray]]:
    """2D planes for the UI, using the same view helpers as Phase 6."""
    pack: dict[str, dict[str, np.ndarray]] = {}
    for orientation in ORIENTATION_ORDER:
        idx = None if not indices else indices.get(orientation)
        item: dict[str, np.ndarray] = {
            "mri": mri_plane(image, orientation, idx),
            "pred": labels_to_rgb(extract_slice(prediction, orientation, idx)),  # type: ignore[arg-type]
        }
        if target is not None:
            pred_sl = extract_slice(prediction, orientation, idx)  # type: ignore[arg-type]
            true_sl = extract_slice(target, orientation, idx)  # type: ignore[arg-type]
            item["gt"] = labels_to_rgb(true_sl)
            item["error"] = error_rgb(pred_sl, true_sl, region="WT")
        if entropy is not None:
            item["entropy"] = extract_slice(entropy, orientation, idx)  # type: ignore[arg-type]
        if cam is not None:
            item["cam"] = extract_slice(cam, orientation, idx)  # type: ignore[arg-type]
        if occlusion is not None:
            item["occlusion"] = extract_slice(occlusion, orientation, idx)  # type: ignore[arg-type]
        if mc_entropy is not None:
            item["mc_entropy"] = extract_slice(mc_entropy, orientation, idx)  # type: ignore[arg-type]
        pack[orientation] = item
    return pack


def axis_length(prediction: np.ndarray, orientation: str) -> int:
    return int(prediction.shape[ORIENTATION_AXIS[orientation]])  # type: ignore[index]


def bundle_zip(
    image: np.ndarray,
    result: dict[str, Any],
    *,
    target: np.ndarray | None = None,
    out_dir: Path,
) -> bytes:
    """Zip research PNGs + JSON. Contents come from the inference dict."""
    from neurovisionx.explainability.gradcam import GradCAMResult

    out_dir.mkdir(parents=True, exist_ok=True)
    title = result.get("status_banner", DISCLAIMER)
    cam = result.get("cam")
    if cam is None:
        cam = np.zeros(result["prediction"].shape, dtype=np.float32)
    dummy = GradCAMResult(
        prediction=torch.from_numpy(np.asarray(result["prediction"])),
        logits=torch.from_numpy(np.asarray(result["logits"])),
        cam=torch.from_numpy(np.asarray(cam)),
        class_index=int(result.get("class_index", 3)),
        layer_name="demo_export",
        score=0.0,
        score_mode="export",
    )
    write_explain_panel(
        image,
        dummy,
        out_dir,
        target=target,
        entropy=result.get("entropy"),
        title=title,
    )
    sidecar = {
        "elapsed_ms": result.get("elapsed_ms"),
        "n_params": result.get("n_params"),
        "smoke": result.get("smoke"),
        "device": result.get("device"),
        "patch": result.get("patch"),
        "cam_note": result.get("cam_note"),
        "occlusion_note": result.get("occlusion_note"),
        "mc_note": result.get("mc_note"),
        "class_hist": result.get("class_hist"),
        "scores": result.get("scores"),
        "disclaimer": result.get("disclaimer"),
        "status_banner": result.get("status_banner"),
    }
    (out_dir / "inference.json").write_text(json.dumps(sidecar, indent=2, default=str), encoding="utf-8")
    if result.get("scores"):
        from neurovisionx.evaluation.tables import scores_to_csv_text

        (out_dir / "metrics.csv").write_text(scores_to_csv_text(result["scores"]), encoding="utf-8")
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as zf:
        for path in out_dir.iterdir():
            if path.is_file():
                zf.write(path, arcname=path.name)
    return buffer.getvalue()
