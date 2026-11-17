"""Training loop used only after official BraTS folders are present.

A run is not 'complete' until this function finishes and writes metrics from
real batches. Importing this module is lazy so CPU-only installs can still
surface MissingBraTSDataError.
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


def run_training(request: TrainRequest) -> Path:
    torch = _require_torch()
    from torch.utils.data import DataLoader

    from neurovisionx.data.torch_dataset import BraTSPatchDataset
    from neurovisionx.losses.segmentation import CombinedSegLoss
    from neurovisionx.models.factory import build_model

    data_cfg = request.config["data"]
    model_cfg = request.config["model"]
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
        train_ds,
        batch_size=int(train_cfg["schedule"]["batch_size"]),
        shuffle=True,
        num_workers=0,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=1,
        shuffle=False,
        num_workers=0,
    )

    device = torch.device(
        "cuda" if train_cfg.get("device", "auto") != "cpu" and torch.cuda.is_available() else "cpu"
    )
    model = build_model(model_cfg).to(device)
    loss_cfg = train_cfg["loss"]
    criterion = CombinedSegLoss(
        dice_weight=float(loss_cfg["dice_weight"]),
        ce_weight=float(loss_cfg["ce_weight"]),
        boundary_weight=float(loss_cfg["boundary_weight"]),
        smooth=float(loss_cfg["smooth"]),
        include_background=bool(loss_cfg["include_background"]),
    )
    optim = torch.optim.AdamW(
        model.parameters(),
        lr=float(train_cfg["optim"]["lr"]),
        weight_decay=float(train_cfg["optim"]["weight_decay"]),
    )

    history: list[dict[str, float]] = []
    best_val = float("inf")
    best_path = request.run_dir / "best.pt"
    epochs = int(train_cfg["schedule"]["epochs"])
    use_amp = bool(train_cfg["schedule"].get("amp", False)) and device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    for epoch in range(1, epochs + 1):
        model.train()
        train_losses: list[float] = []
        for batch in train_loader:
            images = batch["image"].to(device)
            labels = batch["label"].to(device)
            optim.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=use_amp):
                logits = model(images)
                distance_map = None
                if criterion.boundary_weight > 0:
                    from neurovisionx.data.distance import one_hot_distance_map

                    distance_map = one_hot_distance_map(labels, logits.shape[1])
                loss = criterion(logits, labels, distance_map=distance_map)
            scaler.scale(loss).backward()
            scaler.step(optim)
            scaler.update()
            train_losses.append(float(loss.item()))

        model.eval()
        val_losses: list[float] = []
        with torch.no_grad():
            for batch in val_loader:
                images = batch["image"].to(device)
                labels = batch["label"].to(device)
                logits = model(images)
                distance_map = None
                if criterion.boundary_weight > 0:
                    from neurovisionx.data.distance import one_hot_distance_map

                    distance_map = one_hot_distance_map(labels, logits.shape[1])
                val_losses.append(float(criterion(logits, labels, distance_map=distance_map).item()))

        row = {
            "epoch": float(epoch),
            "train_loss": sum(train_losses) / len(train_losses),
            "val_loss": sum(val_losses) / len(val_losses),
        }
        history.append(row)
        if row["val_loss"] < best_val:
            best_val = row["val_loss"]
            torch.save(
                {
                    "model": model.state_dict(),
                    "epoch": epoch,
                    "val_loss": best_val,
                    "config": request.config,
                },
                best_path,
            )

        if epoch % int(train_cfg["logging"]["save_every_epoch"]) == 0:
            torch.save(model.state_dict(), request.run_dir / f"epoch_{epoch:03d}.pt")

    metrics_path = request.run_dir / "history.json"
    metrics_path.write_text(json.dumps(history, indent=2), encoding="utf-8")
    write_metadata(
        request.run_dir / "metadata.json",
        build_run_metadata(
            request.config,
            {
                "status": "completed",
                "epochs_ran": epochs,
                "best_val_loss": best_val,
                "device": str(device),
                "n_train": len(split.train),
                "n_val": len(split.val),
                "n_test": len(split.test),
            },
        ),
    )
    return best_path
