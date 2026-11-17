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
from neurovisionx.evaluation.regions import region_scores
from neurovisionx.utils.config import load_experiment_config
from neurovisionx.utils.paths import project_root


def _mean_std(values: list[float]) -> dict[str, float | None]:
    finite = [v for v in values if v == v]
    if not finite:
        return {"mean": None, "std": None, "n": 0}
    arr = np.asarray(finite, dtype=np.float64)
    return {"mean": float(arr.mean()), "std": float(arr.std(ddof=0)), "n": int(arr.size)}


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
        from neurovisionx.evaluation.predict import infer_volume, load_model_from_checkpoint
    except ImportError as exc:
        raise SystemExit('PyTorch is required. pip install -e ".[research]"') from exc

    model, payload = load_model_from_checkpoint(ckpt, config["model"], map_location="cpu")
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
    for patient_id in patient_ids:
        if patient_id in cases:
            files = cases[patient_id].path_map()
        else:
            from neurovisionx.data.io import resolve_patient_files

            files = resolve_patient_files(raw / patient_id)
        channels = [load_nifti(files[name]) for name in ("t1", "t1ce", "t2", "flair")]
        image = zscore_channels(np.stack(channels, axis=0))
        target = remap_seg(load_nifti(files["seg"]))
        volume = torch.from_numpy(np.ascontiguousarray(image)).unsqueeze(0)
        pred = infer_volume(model, volume, patch, overlap=overlap).cpu().numpy()
        scores = region_scores(pred, target, compute_hd95=True)
        scores["patient_id"] = patient_id
        rows.append(scores)

    summary = {}
    for key in ("dice_wt", "dice_tc", "dice_et", "hd95_wt", "hd95_tc", "hd95_et"):
        summary[key] = _mean_std([float(row[key]) for row in rows if key in row])

    out = run_dir / f"metrics_{args.split}.json"
    payload_out = {
        "split": args.split,
        "n_patients": len(rows),
        "patients": rows,
        "summary": summary,
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
