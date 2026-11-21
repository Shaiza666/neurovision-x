"""Download or copy the published MSD CPU checkpoints and check SHA-256.

Reads configs/checkpoints.yaml. Each run is written to
experiments/<run>/best.pt (or --dest/<run>/<filename>). A digest mismatch
deletes that file and exits non-zero.

These weights are an experimental research prototype. They are not a
clinically validated diagnostic system.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from neurovisionx.demo.fetch_weights import CheckpointFetchError, materialize_checkpoint
from neurovisionx.utils.config import load_yaml


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch MSD CPU checkpoints and verify SHA-256.")
    parser.add_argument("--config", default="configs/checkpoints.yaml")
    parser.add_argument("--dest", type=Path, default=ROOT / "experiments")
    args = parser.parse_args()
    payload = load_yaml(args.config)
    print(payload.get("clinical_disclaimer", ""))
    runs = payload.get("runs") or {}
    if not runs:
        raise SystemExit("No runs listed in the checkpoint config.")
    failures = 0
    for run_name, spec in runs.items():
        if not isinstance(spec, dict):
            print(f"{run_name}: spec is not a mapping")
            failures += 1
            continue
        filename = str(spec.get("filename") or "best.pt")
        target = args.dest / str(run_name) / filename
        try:
            kind = materialize_checkpoint(spec, target, repo_root=ROOT)
        except CheckpointFetchError as exc:
            print(f"{run_name}: FAILED {exc}")
            failures += 1
            continue
        print(f"{run_name}: {kind} {target}")
    if failures:
        raise SystemExit(f"{failures} checkpoint(s) failed SHA-256 placement.")
    print("All listed checkpoints match configs/checkpoints.yaml.")


if __name__ == "__main__":
    main()
