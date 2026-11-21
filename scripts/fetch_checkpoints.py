"""Fetch published checkpoints, or print how to reproduce them.

This script does not upload weights. configs/checkpoints.yaml records an
optional URL and sha256 for each run. A null URL means the weights are not
hosted and must be trained locally. Checkpoints are gitignored.
"""

from __future__ import annotations

import argparse
import hashlib
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from neurovisionx.utils.config import load_yaml


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch or verify MSD CPU checkpoints.")
    parser.add_argument("--config", default="configs/checkpoints.yaml")
    parser.add_argument("--dest", type=Path, default=ROOT / "experiments")
    args = parser.parse_args()
    payload = load_yaml(args.config)
    runs = payload.get("runs") or {}
    print(payload.get("clinical_disclaimer", ""))
    if not payload.get("published"):
        print(payload.get("reason", "Weights are not published."))
    any_url = False
    for run_name, spec in runs.items():
        url = spec.get("url")
        expected = spec.get("sha256")
        filename = spec.get("filename") or "best.pt"
        target = args.dest / run_name / filename
        if url:
            any_url = True
            target.parent.mkdir(parents=True, exist_ok=True)
            print(f"Downloading {run_name} from {url}")
            urllib.request.urlretrieve(url, target)
        if target.is_file() and expected:
            actual = _sha256(target)
            status = "ok" if actual == expected else "MISMATCH"
            print(f"{run_name}: sha256 {status} {actual}")
        elif target.is_file():
            print(f"{run_name}: local file {target} sha256={_sha256(target)} (no expected digest in config)")
        elif not url:
            print(f"{run_name}: no URL and no local file at {target}")
    if not any_url:
        print()
        print("Reproduce the reduced CPU experiment (MSD Task01, not official BraTS):")
        print("  python scripts/download_msd_task01.py")
        print("  python scripts/prepare_msd_cpu_cohort.py")
        print("  python scripts/train.py --ablation A0_baseline --run-name A0_baseline \\")
        print("      --data-config configs/data_msd_cpu.yaml --model-config configs/model_msd_cpu.yaml \\")
        print("      --training-config configs/training_msd_cpu.yaml")
        print("  python scripts/train.py --ablation A2_attention --run-name A2_attention \\")
        print("      --data-config configs/data_msd_cpu.yaml --model-config configs/model_msd_cpu.yaml \\")
        print("      --training-config configs/training_msd_cpu.yaml")
        print("  python scripts/train.py --ablation A5_combination --run-name A5_combination \\")
        print("      --data-config configs/data_msd_cpu.yaml --model-config configs/model_msd_cpu.yaml \\")
        print("      --training-config configs/training_msd_cpu.yaml")
        print("  python scripts/evaluate.py --run-name A0_baseline")
        print("  python scripts/evaluate.py --run-name A2_attention")
        print("  python scripts/evaluate.py --run-name A5_combination")


if __name__ == "__main__":
    main()
