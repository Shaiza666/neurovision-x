"""Run sliding-window inference from a checkpoint.

Refuses smoke checkpoints when --require-real is set. Without real NIfTI,
this script only checks that a checkpoint loads (shape test).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from neurovisionx.utils.config import load_experiment_config
from neurovisionx.utils.paths import project_root


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-name", default="debug")
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--allow-smoke", action="store_true")
    args = parser.parse_args()
    config = load_experiment_config()
    run_dir = project_root() / config["data"]["paths"]["output_root"] / args.run_name
    ckpt = args.checkpoint or (run_dir / "best.pt")
    if not ckpt.is_file():
        latest = run_dir / "latest.pt"
        ckpt = latest if latest.is_file() else ckpt
    if not ckpt.is_file():
        raise SystemExit(f"No checkpoint at {ckpt}. Inference was not run.")

    try:
        import torch
    except ImportError as exc:
        raise SystemExit('PyTorch is required. pip install -e ".[research]"') from exc

    from neurovisionx.evaluation.predict import infer_volume, load_model_from_checkpoint

    model, payload = load_model_from_checkpoint(ckpt, config["model"], map_location="cpu")
    if payload.get("smoke") and not args.allow_smoke:
        raise SystemExit("Smoke checkpoint refused. Pass --allow-smoke for a tensor shape check only.")
    patch = tuple(int(v) for v in config["training"].get("smoke", {}).get("patch_size", config["training"]["schedule"]["patch_size"]))
    dummy = torch.zeros(1, int(config["model"]["in_channels"]), *patch)
    pred = infer_volume(model, dummy, patch)
    tag = "SMOKE shape check — not research" if payload.get("smoke") else "inference shape check"
    print(f"{tag}: pred {tuple(pred.shape)} from {ckpt}")


if __name__ == "__main__":
    main()
