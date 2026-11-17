"""Train on official BraTS volumes, or run a labeled synthetic smoke test."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from neurovisionx.training.loop import train_or_explain


def main() -> None:
    parser = argparse.ArgumentParser(description="Train 3D segmentation (BraTS or --smoke).")
    parser.add_argument("--run-name", default="debug")
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Synthetic tensor smoke test. Not BraTS. Not research results.",
    )
    parser.add_argument("--resume", action="store_true", help="Resume from experiments/<run>/latest.pt")
    parser.add_argument(
        "--ablation",
        default=None,
        help="Apply a configs/ablations.yaml arm. Does not skip the BraTS data gate. Do not run until the protocol is approved.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Override training.yaml seed for a multi-seed job. Does not skip the data gate.",
    )
    args = parser.parse_args()
    train_or_explain(
        run_name=args.run_name,
        smoke=args.smoke,
        resume=args.resume,
        ablation_id=args.ablation,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
