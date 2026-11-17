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
from neurovisionx.data.io import load_nifti, resolve_patient_files
from neurovisionx.data.preprocess import remap_seg, zscore_channels
from neurovisionx.data.splits import load_split
from neurovisionx.evaluation.regions import region_scores
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
    except ImportError as exc:
        raise SystemExit('PyTorch is required. pip install -e ".[research]"') from exc

    from neurovisionx.evaluation.inference import sliding_window_logits
    from neurovisionx.models.factory import build_model

    payload = torch.load(ckpt, map_location="cpu")
    state = payload["model"] if isinstance(payload, dict) and "model" in payload else payload
    model = build_model(config["model"])
    model.load_state_dict(state)
    model.eval()
    patch = tuple(int(v) for v in config["training"]["schedule"]["patch_size"])

    split_path = run_dir / "patients.json"
    if split_path.is_file():
        split = load_split(split_path)
        patient_ids = list(getattr(split, args.split))
    else:
        patient_ids = list_patient_ids(raw)

    rows = []
    for patient_id in patient_ids:
        paths = resolve_patient_files(raw / patient_id)
        channels = [load_nifti(paths[name]) for name in ("t1", "t1ce", "t2", "flair")]
        image = zscore_channels(np.stack(channels, axis=0))
        target = remap_seg(load_nifti(paths["seg"]))
        volume = torch.from_numpy(image).unsqueeze(0)
        logits = sliding_window_logits(model, volume, patch)
        pred = logits.argmax(dim=1)[0].cpu().numpy()
        scores = region_scores(pred, target, compute_hd95=True)
        scores["patient_id"] = patient_id
        rows.append(scores)

    out = run_dir / f"metrics_{args.split}.json"
    out.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(f"Wrote {out} ({len(rows)} patients). Research prototype — not a clinical report.")


if __name__ == "__main__":
    main()
