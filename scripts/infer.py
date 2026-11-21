"""Run sliding-window inference from a checkpoint.

Refuses smoke checkpoints when --require-real is set. Without real NIfTI,
this script only checks that a checkpoint loads (shape test). Pass --case-dir
to run on an on-disk four-modality folder; missing files are errors, not zeros.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from neurovisionx.utils.paths import project_root


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-name", default="debug")
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--allow-smoke", action="store_true")
    parser.add_argument(
        "--case-dir",
        type=Path,
        default=None,
        help="Folder with co-registered T1/T1ce/T2/FLAIR NIfTI. Not synthesized if missing.",
    )
    args = parser.parse_args()
    from neurovisionx.utils.config import load_experiment_config

    config = load_experiment_config()
    run_dir = project_root() / config["data"]["paths"]["output_root"] / args.run_name
    ckpt = args.checkpoint or (run_dir / "best.pt")
    if not ckpt.is_file():
        latest = run_dir / "latest.pt"
        ckpt = latest if latest.is_file() else ckpt
    if not ckpt.is_file():
        raise SystemExit(f"No checkpoint at {ckpt}. Inference was not run.")

    try:
        import numpy as np
        import torch
    except ImportError as exc:
        raise SystemExit('PyTorch is required. pip install -e ".[research]"') from exc

    from neurovisionx.demo.checkpoints import load_demo_model, patch_size_from_payload
    from neurovisionx.evaluation.predict import infer_volume

    model, payload = load_demo_model(ckpt, map_location="cpu")
    if payload.get("smoke") and not args.allow_smoke:
        raise SystemExit("Smoke checkpoint refused. Pass --allow-smoke for a tensor shape check only.")
    patch = patch_size_from_payload(payload)

    if args.case_dir is None:
        dummy = torch.zeros(1, int(payload.get("config", {}).get("model", {}).get("in_channels", 4)), *patch)
        pred = infer_volume(model, dummy, patch)
        tag = "SMOKE shape check — not research" if payload.get("smoke") else "inference shape check"
        print(f"{tag}: pred {tuple(pred.shape)} from {ckpt}")
        return

    from neurovisionx.data.errors import MissingBraTSDataError
    from neurovisionx.data.io import IMAGE_MODALITIES, find_modality, load_nifti
    from neurovisionx.data.preprocess import zscore_channels

    case_dir = args.case_dir if args.case_dir.is_absolute() else (project_root() / args.case_dir)
    try:
        files = {name: find_modality(case_dir, name) for name in IMAGE_MODALITIES}
    except MissingBraTSDataError as exc:
        raise SystemExit(f"{exc} Empty channels are not synthesized.") from exc
    channels = [load_nifti(files[name]) for name in IMAGE_MODALITIES]
    image = zscore_channels(np.stack(channels, axis=0))
    volume = torch.from_numpy(np.ascontiguousarray(image, dtype=np.float32)).unsqueeze(0)
    pred = infer_volume(model, volume, patch)
    unique, counts = np.unique(pred.detach().cpu().numpy(), return_counts=True)
    hist = {int(k): int(v) for k, v in zip(unique, counts)}
    tag = "SMOKE inference — not research" if payload.get("smoke") else "checkpoint inference"
    print(f"{tag}: pred {tuple(pred.shape)} class_hist={hist} from {ckpt}")
    print("No Dice/HD95 printed: upload or pass a segmentation through evaluate.py after licensed BraTS data.")


if __name__ == "__main__":
    main()
