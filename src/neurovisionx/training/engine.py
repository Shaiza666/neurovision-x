"""Training loop shared by official BraTS jobs and the synthetic smoke test.

A BraTS run is not 'complete' until this function finishes on real volumes.
Smoke history is labeled synthetic and must not be reported as research.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from neurovisionx.data.dataset import list_complete_cases
from neurovisionx.data.splits import save_split, split_from_cases
from neurovisionx.training.loop import TrainRequest
from neurovisionx.utils.metadata import build_run_metadata, write_metadata


def _require_torch() -> Any:
    try:
        import torch
    except ImportError as exc:
        raise SystemExit(
            'PyTorch is required to train. Install with: pip install -e ".[research]"'
        ) from exc
    return torch


def _build_criterion(train_cfg: dict[str, Any], torch: Any):
    from neurovisionx.losses.segmentation import CombinedSegLoss

    loss_cfg = train_cfg["loss"]
    weights = loss_cfg.get("class_weights")
    return CombinedSegLoss(
        dice_weight=float(loss_cfg["dice_weight"]),
        ce_weight=float(loss_cfg["ce_weight"]),
        boundary_weight=float(loss_cfg["boundary_weight"]),
        smooth=float(loss_cfg["smooth"]),
        include_background=bool(loss_cfg["include_background"]),
        class_weight_mode=str(loss_cfg.get("class_weight_mode", "none")),
        class_weights=weights,
        boundary_classes=loss_cfg.get("boundary_classes"),
    )


def _metric_value(row: dict[str, float], name: str, default: float) -> float:
    if name in row and row[name] == row[name]:
        return float(row[name])
    return default


def fit(
    request: TrainRequest,
    train_loader: Any,
    val_loader: Any,
    *,
    smoke: bool = False,
    resume: bool = False,
    model_cfg: dict[str, Any] | None = None,
    epochs_override: int | None = None,
) -> Path:
    torch = _require_torch()
    from neurovisionx.models.factory import build_model
    from neurovisionx.training.amp_util import amp_enabled, autocast, make_scaler
    from neurovisionx.training.checkpoint import load_checkpoint, restore_training_state, save_checkpoint
    from neurovisionx.training.curves import save_learning_curves
    from neurovisionx.training.validate import _maybe_distance, validate_epoch

    train_cfg = request.config["training"]
    cfg_model = model_cfg or request.config["model"]
    device = torch.device(
        "cuda" if train_cfg.get("device", "auto") != "cpu" and torch.cuda.is_available() else "cpu"
    )
    model = build_model(cfg_model).to(device)
    from neurovisionx.evaluation.cost import count_parameters

    n_params = count_parameters(model)
    criterion = _build_criterion(train_cfg, torch)
    optim = torch.optim.AdamW(
        model.parameters(),
        lr=float(train_cfg["optim"]["lr"]),
        weight_decay=float(train_cfg["optim"]["weight_decay"]),
    )
    scheduler = None
    sched_name = str(train_cfg.get("optim", {}).get("scheduler", "none")).lower()
    if sched_name in {"plateau", "reducelronplateau"}:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optim,
            mode=str(train_cfg.get("early_stopping", {}).get("mode", "max")),
            factor=float(train_cfg.get("optim", {}).get("scheduler_factor", 0.5)),
            patience=int(train_cfg.get("optim", {}).get("scheduler_patience", 5)),
        )
    use_amp = amp_enabled(train_cfg, device)
    scaler = make_scaler(use_amp)
    grad_clip = float(train_cfg.get("grad_clip", 0.0))
    validate_every = int(train_cfg.get("validate_every", 1))
    hd95_val = bool(train_cfg.get("compute_hd95_on_val", False))
    stop_cfg = train_cfg.get("early_stopping", {})
    monitor = str(stop_cfg.get("metric", "val_loss"))
    mode = str(stop_cfg.get("mode", "min"))
    patience = int(stop_cfg.get("patience", 15))
    best_metric = float("-inf") if mode == "max" else float("inf")
    history: list[dict[str, float]] = []
    start_epoch = 1
    best_path = request.run_dir / "best.pt"
    latest_path = request.run_dir / "latest.pt"
    epochs = int(epochs_override or train_cfg["schedule"]["epochs"])

    if resume and latest_path.is_file():
        payload = load_checkpoint(latest_path, map_location=device)
        start_epoch = restore_training_state(payload, model, optim, scaler)
        history = list(payload.get("history", []))
        best_metric = float(payload.get("best_metric", best_metric))

    epochs_without_improve = 0
    for epoch in range(start_epoch, epochs + 1):
        model.train()
        train_losses: list[float] = []
        for batch in train_loader:
            images = batch["image"].to(device)
            labels = batch["label"].to(device)
            optim.zero_grad(set_to_none=True)
            with autocast(device, use_amp):
                logits = model(images)
                loss = criterion(logits, labels, distance_map=_maybe_distance(criterion, logits, labels))
            scaler.scale(loss).backward()
            if grad_clip > 0:
                scaler.unscale_(optim)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optim)
            scaler.update()
            train_losses.append(float(loss.item()))

        row: dict[str, float] = {
            "epoch": float(epoch),
            "train_loss": sum(train_losses) / max(len(train_losses), 1),
        }
        do_val = epoch % validate_every == 0 or epoch == epochs
        if do_val:
            val_row = validate_epoch(
                model, val_loader, criterion, device, compute_hd95=hd95_val
            )
            row.update(val_row)
            current = _metric_value(row, monitor, _metric_value(row, "val_loss", row["train_loss"]))
            improved = current > best_metric if mode == "max" else current < best_metric
            if improved:
                best_metric = current
                epochs_without_improve = 0
                save_checkpoint(
                    best_path,
                    model=model,
                    optimizer=optim,
                    scaler=scaler,
                    epoch=epoch,
                    history=history + [row],
                    config=request.config,
                    best_metric=best_metric,
                    extra={"smoke": smoke, "kind": "best"},
                )
            else:
                epochs_without_improve += 1
            if scheduler is not None:
                scheduler.step(current)
        history.append(row)
        save_checkpoint(
            latest_path,
            model=model,
            optimizer=optim,
            scaler=scaler,
            epoch=epoch,
            history=history,
            config=request.config,
            best_metric=best_metric,
            extra={"smoke": smoke, "kind": "latest"},
        )
        if patience > 0 and epochs_without_improve >= patience:
            break

    (request.run_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    title = (
        "SMOKE synthetic curves — not research results"
        if smoke
        else "Research learning curves — not clinical performance"
    )
    try:
        save_learning_curves(history, request.run_dir / "learning_curves.png", title=title)
    except ImportError:
        pass
    write_metadata(
        request.run_dir / "metadata.json",
        build_run_metadata(
            request.config,
            {
                "status": "smoke_synthetic_not_research" if smoke else "completed",
                "smoke": smoke,
                "note": (
                    "Synthetic tensors only. Not BraTS. Not research results."
                    if smoke
                    else str(
                        request.config.get("data", {}).get("dataset", {}).get("run_note")
                        or "Training finished on the data that was present for this run."
                    )
                ),
                "epochs_ran": history[-1]["epoch"] if history else 0,
                "best_metric": best_metric,
                "monitor": monitor,
                "device": str(device),
                "amp": use_amp,
                "n_params": n_params,
            },
        ),
    )
    return best_path if best_path.is_file() else latest_path


def run_training(request: TrainRequest, *, resume: bool = False) -> Path:
    torch = _require_torch()
    from torch.utils.data import DataLoader

    from neurovisionx.data.torch_dataset import BraTSPatchDataset

    data_cfg = request.config["data"]
    train_cfg = request.config["training"]
    raw = request.data_root
    cases = list_complete_cases(raw)
    if not cases:
        raise SystemExit(f"No complete labeled patients under {raw}. No training started.")
    case_map = {case.patient_id: case for case in cases}
    split_cfg = data_cfg["splits"]
    split = split_from_cases(
        cases,
        train_ratio=float(split_cfg["train_ratio"]),
        val_ratio=float(split_cfg["val_ratio"]),
        test_ratio=float(split_cfg["test_ratio"]),
        seed=int(split_cfg["seed"]),
        honor_official_validation=bool(split_cfg.get("honor_official_validation", True)),
    )
    save_split(split, request.run_dir / "patients.json")
    patch = tuple(int(v) for v in train_cfg["schedule"]["patch_size"])
    spacing = tuple(float(v) for v in data_cfg["preprocessing"]["target_spacing_mm"])
    crop_fg = bool(data_cfg["preprocessing"].get("crop_foreground", True))
    fg_prob = float(data_cfg.get("patches", {}).get("foreground_prob", 0.67))
    train_ds = BraTSPatchDataset(
        raw,
        split.train,
        patch,
        random_crop=True,
        seed=int(train_cfg["seed"]),
        cases=case_map,
        target_spacing_mm=spacing,
        crop_foreground=crop_fg,
        foreground_prob=fg_prob,
    )
    val_ds = BraTSPatchDataset(
        raw,
        split.val,
        patch,
        random_crop=False,
        seed=int(train_cfg["seed"]),
        cases=case_map,
        target_spacing_mm=spacing,
        crop_foreground=crop_fg,
        foreground_prob=fg_prob,
    )
    train_loader = DataLoader(
        train_ds, batch_size=int(train_cfg["schedule"]["batch_size"]), shuffle=True, num_workers=0
    )
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=0)
    return fit(request, train_loader, val_loader, smoke=False, resume=resume)


def run_smoke(run_name: str = "smoke") -> Path:
    from neurovisionx.training.smoke import prepare_smoke_request, smoke_loaders, smoke_model_cfg

    request = prepare_smoke_request(run_name)
    train_loader, val_loader = smoke_loaders(request.config)
    epochs = int(request.config["training"].get("smoke", {}).get("epochs", 2))
    model_cfg = smoke_model_cfg(request.config)
    return fit(
        request,
        train_loader,
        val_loader,
        smoke=True,
        resume=False,
        model_cfg=model_cfg,
        epochs_override=epochs,
    )
