"""Evaluate a checkpoint. Refuses to invent Dice/HD95 without real data."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from neurovisionx.data.dataset import MissingBraTSDataError, list_patient_ids, require_brats_root
from neurovisionx.data.inventory import discover_inventory
from neurovisionx.data.io import load_nifti
from neurovisionx.data.preprocess import remap_seg, zscore_channels
from neurovisionx.data.splits import load_split
from neurovisionx.evaluation.metrics import (
    brier_score_multiclass,
    expected_calibration_error,
    negative_log_likelihood,
)
from neurovisionx.evaluation.regions import region_scores
from neurovisionx.evaluation.summarize import summarize_values
from neurovisionx.evaluation.tables import write_patient_csv
from neurovisionx.evaluation.run_config import config_from_checkpoint
from neurovisionx.utils.config import load_experiment_config
from neurovisionx.utils.paths import project_root


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-name", default="debug")
    parser.add_argument("--data-config", default="configs/data.yaml")
    parser.add_argument("--model-config", default="configs/model.yaml")
    parser.add_argument("--training-config", default="configs/training.yaml")
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--split", default="test", choices=("train", "val", "test"))
    parser.add_argument(
        "--save-preds",
        action="store_true",
        help="Write prediction NIfTI under experiments/<run>/preds/. Not uploaded to git.",
    )
    args = parser.parse_args()

    config = load_experiment_config(args.data_config, args.model_config, args.training_config)
    root = project_root()
    run_dir = root / config["data"]["paths"]["output_root"] / args.run_name
    ckpt = args.checkpoint or (run_dir / "best.pt")
    payload = None
    if ckpt.is_file():
        from neurovisionx.training.checkpoint import load_checkpoint

        payload = load_checkpoint(ckpt, map_location="cpu")
        config = config_from_checkpoint(payload, config)
    raw = root / config["data"]["paths"]["raw_root"]
    try:
        require_brats_root(raw)
    except MissingBraTSDataError as exc:
        raise SystemExit(str(exc)) from exc
    if not ckpt.is_file():
        raise SystemExit(
            f"No checkpoint at {ckpt}. Training has not been run. No metrics were fabricated."
        )

    try:
        import torch
        from neurovisionx.evaluation.cost import count_parameters, measure_gpu_memory, time_inference
        from neurovisionx.evaluation.inference import sliding_window_logits
        from neurovisionx.evaluation.predict import load_model_from_checkpoint
    except ImportError as exc:
        raise SystemExit('PyTorch is required. pip install -e ".[research]"') from exc

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, payload = load_model_from_checkpoint(ckpt, config["model"], map_location=device)
    config = config_from_checkpoint(payload, config)
    model = model.to(device)
    if payload.get("smoke"):
        raise SystemExit(
            "This checkpoint was written by a synthetic smoke test. "
            "It is not a BraTS result and will not be evaluated as one."
        )
    patch = tuple(int(v) for v in config["training"]["schedule"]["patch_size"])
    overlap = float(config["training"]["schedule"].get("infer_overlap", 0.5))
    from neurovisionx.uncertainty.tta import normalize_uncertainty_method

    uncertainty = config["training"].get("uncertainty") or {}
    uncertainty_method = normalize_uncertainty_method(uncertainty.get("method", "none"))
    mc_samples = int(uncertainty.get("samples") or 1)
    use_mc = uncertainty_method == "mc_dropout" and mc_samples > 1

    split_path = run_dir / "patients.json"
    if split_path.is_file():
        split = load_split(split_path)
        patient_ids = list(getattr(split, args.split))
    else:
        patient_ids = list_patient_ids(raw)

    inventory = discover_inventory(raw)
    cases = inventory.by_id()
    rows = []
    timed = None
    gpu_mem = None
    for patient_id in patient_ids:
        if patient_id in cases:
            files = cases[patient_id].path_map()
        else:
            from neurovisionx.data.io import resolve_patient_files

            files = resolve_patient_files(raw / patient_id)
        channels = [load_nifti(files[name]) for name in ("t1", "t1ce", "t2", "flair")]
        image = zscore_channels(np.stack(channels, axis=0))
        target = remap_seg(load_nifti(files["seg"]))
        volume = torch.from_numpy(np.ascontiguousarray(image)).unsqueeze(0).to(device)
        if use_mc:
            from neurovisionx.evaluation.predict import infer_volume_mc

            mc = infer_volume_mc(model, volume, patch, overlap=overlap, samples=mc_samples)
            probs = mc["mean_probs"].detach().cpu().numpy()
            pred = mc["prediction"].detach().cpu().numpy()
        else:
            logits = sliding_window_logits(model, volume, patch, overlap=overlap)
            probs = torch.softmax(logits, dim=1)[0].detach().cpu().numpy()
            pred = probs.argmax(axis=0)
        scores = region_scores(pred, target, compute_hd95=True, compute_pr=True)
        scores["brier"] = brier_score_multiclass(probs, target)
        scores["nll"] = negative_log_likelihood(probs, target)
        scores["ece"] = expected_calibration_error(
            probs.max(axis=0), (pred == target).astype(np.float64)
        )
        scores["patient_id"] = patient_id
        rows.append(scores)
        if args.save_preds:
            from neurovisionx.data.io import save_nifti

            pred_dir = run_dir / "preds"
            save_nifti(
                pred_dir / f"{patient_id}_pred.nii.gz",
                pred.astype(np.int16),
                reference=files.get("t1ce") or files.get("t1"),
            )
        if timed is None:
            patch_volume = volume[:, :, : patch[0], : patch[1], : patch[2]]
            timed = time_inference(model, patch_volume, mc_samples=1, n_repeat=3)
            gpu_mem = measure_gpu_memory(model, patch_volume, mc_samples=1, device=device)
            timed["n_params"] = count_parameters(model)

    summary = {}
    metric_keys = (
        "dice_wt",
        "dice_tc",
        "dice_et",
        "hd95_wt",
        "hd95_tc",
        "hd95_et",
        "precision_wt",
        "recall_wt",
        "precision_tc",
        "recall_tc",
        "precision_et",
        "recall_et",
        "sensitivity_wt",
        "sensitivity_tc",
        "sensitivity_et",
        "specificity_wt",
        "specificity_tc",
        "specificity_et",
        "brier",
        "nll",
        "ece",
    )
    for key in metric_keys:
        summary[key] = summarize_values([float(row[key]) for row in rows if key in row])

    out = run_dir / f"metrics_{args.split}.json"
    payload_out = {
        "split": args.split,
        "n_patients": len(rows),
        "patients": rows,
        "summary": summary,
        "cost": timed,
        "gpu_memory": gpu_mem,
        "hd95_note": (
            "HD95 is the 95th percentile of bidirectional surface distances "
            "(see docs/metrics.md). NaN means a surface could not be formed. "
            "Research prototype — not a clinical report."
        ),
        "clinical_disclaimer": "Research prototype. Not a clinically validated diagnostic tool.",
        "inference": {
            "method": "mc_dropout" if use_mc else "deterministic",
            "mc_samples": mc_samples if use_mc else 1,
        },
        "spatial": {
            "hd95_unit": "voxels",
            "target_spacing_mm": list(config["data"]["preprocessing"].get("target_spacing_mm", [])),
            "note": (
                "HD95 is in voxels of the volume that was scored. "
                "Do not relabel it as millimetres unless that spacing is 1 mm."
            ),
        },
        "dataset_name": (config.get("data") or {}).get("dataset", {}).get("name"),
    }
    out.write_text(json.dumps(payload_out, indent=2), encoding="utf-8")
    csv_path = write_patient_csv(rows, run_dir / f"metrics_{args.split}.csv")
    print(f"Wrote {out} and {csv_path} ({len(rows)} patients). Research prototype — not a clinical report.")


if __name__ == "__main__":
    main()
