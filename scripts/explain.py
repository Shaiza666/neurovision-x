"""Write Grad-CAM and overlay figures from an actual model pass.

Refuses to invent BraTS cases. --synthetic writes a watermarked demo on
random tensors (not MRI, not research). Smoke checkpoints are refused
unless --allow-smoke (still watermarked).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from neurovisionx.data.dataset import MissingBraTSDataError, require_brats_root
from neurovisionx.explainability.gradcam import DISCLAIMER
from neurovisionx.utils.config import load_experiment_config
from neurovisionx.utils.paths import project_root

SYNTHETIC_TITLE = (
    "SYNTHETIC RANDOM TENSORS — not MRI, not a patient, not research. " + DISCLAIMER
)


def _synthetic_panel(out_dir: Path, class_index: int, also_gradient: bool) -> None:
    import torch

    from neurovisionx.explainability.panel import explain_volume
    from neurovisionx.models.factory import build_model

    model = build_model(
        {
            "name": "unet3d",
            "in_channels": 4,
            "out_channels": 4,
            "features": [8, 16],
            "dropout": 0.0,
            "optional_monai": False,
            "fusion": "early",
        }
    )
    torch.manual_seed(2026)
    volume = torch.randn(1, 4, 16, 16, 16)
    image = volume[0].numpy()
    target = torch.zeros(16, 16, 16, dtype=torch.long)
    target[4:12, 4:12, 4:12] = 3
    explain_volume(
        model,
        volume,
        image,
        out_dir,
        target=target.numpy(),
        class_index=class_index,
        also_gradient=also_gradient,
        title=SYNTHETIC_TITLE,
    )
    print(f"Wrote synthetic (not MRI) panel under {out_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Explainability figures. Not a clinical report."
    )
    parser.add_argument("--run-name", default="debug")
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--patient-id", default=None)
    parser.add_argument("--synthetic", action="store_true", help="Random tensors; not MRI.")
    parser.add_argument("--allow-smoke", action="store_true")
    parser.add_argument("--class-index", type=int, default=3)
    parser.add_argument("--mc-samples", type=int, default=0)
    parser.add_argument("--also-gradient", action="store_true")
    parser.add_argument("--out-dir", type=Path, default=None)
    args = parser.parse_args()
    root = project_root()
    config = load_experiment_config()

    if args.synthetic:
        out = args.out_dir or (root / "docs" / "figures" / "explain_synthetic_not_mri")
        _synthetic_panel(out, args.class_index, args.also_gradient)
        return

    raw = root / config["data"]["paths"]["raw_root"]
    try:
        require_brats_root(raw)
    except MissingBraTSDataError as exc:
        raise SystemExit(str(exc)) from exc

    if not args.patient_id:
        raise SystemExit("Pass --patient-id for a real volume, or --synthetic for a layout demo.")

    run_dir = root / config["data"]["paths"]["output_root"] / args.run_name
    ckpt = args.checkpoint or (run_dir / "best.pt")
    if not ckpt.is_file():
        raise SystemExit(
            f"No checkpoint at {ckpt}. Explainability was not run. No heatmap was fabricated."
        )

    try:
        import numpy as np
        import torch
        from neurovisionx.data.inventory import discover_inventory
        from neurovisionx.data.io import load_nifti, resolve_patient_files
        from neurovisionx.data.preprocess import remap_seg, zscore_channels
        from neurovisionx.evaluation.predict import load_model_from_checkpoint
        from neurovisionx.explainability.panel import explain_volume
        from neurovisionx.uncertainty.mc_dropout import mc_dropout_stats
    except ImportError as exc:
        raise SystemExit('PyTorch is required. pip install -e ".[research]"') from exc

    model, payload = load_model_from_checkpoint(ckpt, config["model"], map_location="cpu")
    if payload.get("smoke") and not args.allow_smoke:
        raise SystemExit(
            "Smoke checkpoint refused. Pass --allow-smoke for a watermarked tensor demo, "
            "or --synthetic. Smoke heatmaps are not BraTS explanations."
        )

    inventory = discover_inventory(raw)
    cases = inventory.by_id()
    if args.patient_id in cases:
        files = cases[args.patient_id].path_map()
    else:
        files = resolve_patient_files(raw / args.patient_id)
    channels = [load_nifti(files[name]) for name in ("t1", "t1ce", "t2", "flair")]
    image = zscore_channels(np.stack(channels, axis=0))
    target = remap_seg(load_nifti(files["seg"])) if "seg" in files else None
    volume = torch.from_numpy(np.ascontiguousarray(image)).unsqueeze(0)
    entropy = None
    if args.mc_samples > 1:
        entropy = mc_dropout_stats(model, volume, samples=int(args.mc_samples))["entropy"].cpu().numpy()
    title = DISCLAIMER
    if payload.get("smoke"):
        title = "SMOKE CHECKPOINT — not BraTS. " + DISCLAIMER
    out = args.out_dir or (run_dir / "explain" / args.patient_id)
    explain_volume(
        model,
        volume,
        image,
        out,
        target=target,
        class_index=args.class_index,
        entropy=entropy,
        also_gradient=args.also_gradient,
        title=title,
    )
    print(f"Wrote {out}. {DISCLAIMER}")


if __name__ == "__main__":
    main()
