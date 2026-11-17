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
from neurovisionx.evaluation.metrics import brier_score_multiclass, expected_calibration_error
from neurovisionx.evaluation.regions import region_scores
from neurovisionx.evaluation.summarize import summarize_values
from neurovisionx.utils.config import load_experiment_config
from neurovisionx.utils.paths import project_root


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-name", default="debug")
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--split", default="test", choices=("train", "val", "test"))
    args = parser.parse_args()

    config = load_experiment_config()
    root = project_root()
    raw = root / config["data"]["paths"]["raw_root"]
    try:
        require_brats_root(raw)
    except MissingBraTSDataError as exc:
        raise SystemExit(str(exc)) from exc

    run_dir = root / config["data"]["paths"]["output_root"] / args.run_name
    ckpt = args.checkpoint or (run_dir / "best.pt")
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
    model = model.to(device)
    if payload.get("smoke"):
        raise SystemExit(
            "This checkpoint was written by a synthetic smoke test. "
            "It is not a BraTS result and will not be evaluated as one."
        )
    patch = tuple(int(v) for v in config["training"]["schedule"]["patch_size"])
    overlap = float(config["training"]["schedule"].get("infer_overlap", 0.5))

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
        logits = sliding_window_logits(model, volume, patch, overlap=overlap)
        probs = torch.softmax(logits, dim=1)[0].detach().cpu().numpy()
        pred = probs.argmax(axis=0)
        scores = region_scores(pred, target, compute_hd95=True, compute_pr=True)
        scores["brier"] = brier_score_multiclass(probs, target)
        scores["ece"] = expected_calibration_error(
            probs.max(axis=0), (pred == target).astype(np.float64)
        )
        scores["patient_id"] = patient_id
        rows.append(scores)
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
        "brier",
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
    }
    out.write_text(json.dumps(payload_out, indent=2), encoding="utf-8")
    print(f"Wrote {out} ({len(rows)} patients). Research prototype — not a clinical report.")


if __name__ == "__main__":
    main()
