"""Calibration, uncertainty–error, cost, and small-ET analysis.

Refuses to invent numbers without official BraTS volumes and a real (non-smoke)
checkpoint. Do not run the expensive MC path until docs/experimental_protocol.md
is approved.
"""

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
from neurovisionx.evaluation.hard_cases import (
    et_voxel_count,
    uncertainty_error_report,
    voxel_error_mask,
)
from neurovisionx.evaluation.metrics import brier_score_multiclass, negative_log_likelihood
from neurovisionx.evaluation.regions import nested_masks, region_scores
from neurovisionx.evaluation.summarize import summarize_values
from neurovisionx.utils.config import load_experiment_config
from neurovisionx.utils.paths import project_root


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-name", default="debug")
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--split", default="test", choices=("train", "val", "test"))
    parser.add_argument("--samples", type=int, default=None, help="MC dropout samples (default: training.yaml)")
    parser.add_argument("--write-figures", action="store_true")
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
        from neurovisionx.evaluation.predict import (
            infer_volume_mc,
            infer_volume_tta,
            load_model_from_checkpoint,
        )
        from neurovisionx.uncertainty.calibration import calibration_summary
        from neurovisionx.uncertainty.mc_dropout import patient_uncertainty_score
    except ImportError as exc:
        raise SystemExit('PyTorch is required. pip install -e ".[research]"') from exc

    model, payload = load_model_from_checkpoint(ckpt, config["model"], map_location="cpu")
    if payload.get("smoke") or payload.get("extra", {}).get("smoke"):
        raise SystemExit(
            "This checkpoint was written by a synthetic smoke test. "
            "It is not a BraTS result and will not be evaluated as one."
        )

    samples = int(args.samples or config["training"]["uncertainty"].get("samples", 10))
    from neurovisionx.uncertainty.tta import normalize_uncertainty_method

    try:
        uncertainty_method = normalize_uncertainty_method(
            config["training"]["uncertainty"].get("method", "mc_dropout")
        )
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
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
    conf_chunks: list[np.ndarray] = []
    correct_chunks: list[np.ndarray] = []
    nlls: list[float] = []
    timed = None
    fig_dir = run_dir / "qc_overlays"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    for index, patient_id in enumerate(patient_ids):
        if patient_id in cases:
            files = cases[patient_id].path_map()
        else:
            from neurovisionx.data.io import resolve_patient_files

            files = resolve_patient_files(raw / patient_id)
        channels = [load_nifti(files[name]) for name in ("t1", "t1ce", "t2", "flair")]
        image = zscore_channels(np.stack(channels, axis=0))
        target = remap_seg(load_nifti(files["seg"]))
        volume = torch.from_numpy(np.ascontiguousarray(image)).unsqueeze(0).to(device)
        if samples <= 1 or uncertainty_method == "none":
            from neurovisionx.evaluation.inference import sliding_window_logits
            from neurovisionx.uncertainty.mc_dropout import predictive_entropy

            logits = sliding_window_logits(model, volume, patch, overlap=overlap)
            probs = torch.softmax(logits, dim=1)[0]
            entropy = predictive_entropy(probs).cpu()
            pred = probs.argmax(dim=0).cpu().numpy()
            probs_np = probs.detach().cpu().numpy()
        elif uncertainty_method == "tta":
            tta = infer_volume_tta(
                model, volume, patch, overlap=overlap, max_views=samples
            )
            pred = tta["prediction"].cpu().numpy()
            entropy = tta["entropy"].cpu()
            probs_np = tta["mean_probs"].cpu().numpy()
        else:
            mc = infer_volume_mc(model, volume, patch, overlap=overlap, samples=samples)
            pred = mc["prediction"].cpu().numpy()
            entropy = mc["entropy"].cpu()
            probs_np = mc["mean_probs"].cpu().numpy()

        scores = region_scores(pred, target, compute_hd95=True, compute_pr=True)
        scores["brier"] = brier_score_multiclass(probs_np, target)
        wt = nested_masks(pred)["WT"] | nested_masks(target)["WT"]
        u_score = patient_uncertainty_score(entropy, torch.from_numpy(wt))
        scores["patient_id"] = patient_id
        scores["et_voxels"] = et_voxel_count(target)
        scores["uncertainty"] = u_score
        rows.append(scores)

        conf = probs_np.max(axis=0)
        correct = (pred == target).astype(np.float64)
        conf_chunks.append(conf.ravel())
        correct_chunks.append(correct.ravel())
        nlls.append(negative_log_likelihood(probs_np, target))

        if timed is None:
            patch_volume = volume[:, :, : patch[0], : patch[1], : patch[2]]
            timed = {
                "t1": time_inference(model, patch_volume, mc_samples=1, n_repeat=3),
                "t_mc": time_inference(model, patch_volume, mc_samples=max(samples, 1), n_repeat=1),
                "gpu_memory_t1": measure_gpu_memory(model, patch_volume, mc_samples=1, device=device),
                "gpu_memory_t_mc": measure_gpu_memory(
                    model, patch_volume, mc_samples=max(samples, 1), device=device
                ),
                "n_params": count_parameters(model),
                "timed_on": "one_patch",
                "note": "Wall-clock and CUDA peak on one patch, not a clinical latency claim.",
            }
        if args.write_figures and index == 0:
            try:
                from neurovisionx.uncertainty.visualize import mid_slice, save_uncertainty_overlay

                save_uncertainty_overlay(
                    mid_slice(image[3]),
                    mid_slice(pred),
                    mid_slice(entropy.numpy()),
                    mid_slice(voxel_error_mask(pred, target)),
                    fig_dir / f"{patient_id}_overlay.png",
                )
            except ImportError:
                pass

    confidence = np.concatenate(conf_chunks) if conf_chunks else np.array([])
    correct = np.concatenate(correct_chunks) if correct_chunks else np.array([])
    cal = calibration_summary(confidence, correct, n_bins=15) if confidence.size else {"ece": None}
    report = uncertainty_error_report(
        np.asarray([row["uncertainty"] for row in rows], dtype=np.float64),
        np.asarray([row["dice_et"] for row in rows], dtype=np.float64),
        np.asarray([row["et_voxels"] for row in rows], dtype=np.float64),
    )
    if args.write_figures and isinstance(cal.get("reliability"), list):
        try:
            from neurovisionx.uncertainty.visualize import save_reliability_diagram

            save_reliability_diagram(cal["reliability"], run_dir / "reliability_diagram.png")
        except ImportError:
            pass

    summary = {}
    for key in (
        "dice_wt",
        "dice_tc",
        "dice_et",
        "hd95_wt",
        "hd95_tc",
        "hd95_et",
        "precision_et",
        "recall_et",
        "brier",
        "uncertainty",
    ):
        summary[key] = summarize_values([float(row[key]) for row in rows if key in row])

    out = run_dir / f"reliability_{args.split}.json"
    payload_out = {
        "split": args.split,
        "n_patients": len(rows),
        "mc_samples": samples,
        "patients": rows,
        "summary": summary,
        "calibration": {
            **{k: v for k, v in cal.items() if k != "reliability"},
            "nll": summarize_values(nlls),
        },
        "reliability_bins": cal.get("reliability"),
        "uncertainty_error": report,
        "cost": timed,
        "clinical_disclaimer": "Research prototype. Not a clinically validated diagnostic tool.",
        "note": (
            "Numbers in this file are research diagnostics from the volumes that "
            "were actually loaded. They are not a clinical report and not a "
            "novelty claim."
        ),
    }
    out.write_text(json.dumps(payload_out, indent=2), encoding="utf-8")
    print(f"Wrote {out} ({len(rows)} patients). Research prototype — not a clinical report.")


if __name__ == "__main__":
    main()
