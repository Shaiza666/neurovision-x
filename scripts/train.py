"""Start training only when official BraTS volumes exist."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from neurovisionx.training.loop import train_or_explain


def main() -> None:
    parser = argparse.ArgumentParser(description="Train 3D U-Net on official BraTS folders.")
    parser.add_argument("--run-name", default="debug")
    args = parser.parse_args()
    train_or_explain(run_name=args.run_name)


if __name__ == "__main__":
    main()
