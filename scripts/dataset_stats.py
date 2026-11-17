"""Class-distribution and shape statistics from loaded segmentations."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from neurovisionx.data.errors import MissingBraTSDataError
from neurovisionx.data.inventory import discover_inventory
from neurovisionx.data.stats import aggregate_stats, patient_stats, write_json
from neurovisionx.data.validate import validate_cases
from neurovisionx.utils.config import load_yaml
from neurovisionx.utils.paths import project_root


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/data.yaml")
    parser.add_argument("--raw-root", type=Path, default=None)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()
    cfg = load_yaml(args.config)
    raw = Path(args.raw_root) if args.raw_root else project_root() / cfg["paths"]["raw_root"]
    try:
        inventory = discover_inventory(raw, max_depth=int(cfg.get("discovery", {}).get("max_depth", 6)))
    except MissingBraTSDataError as exc:
        raise SystemExit(str(exc)) from exc

    report = validate_cases(inventory.cases, require_complete=False, load_voxels=True)
    ok = {row.patient_id for row in report.patients if row.ok or "seg" in inventory.by_id()[row.patient_id].files}
    rows = []
    for case in inventory.cases:
        if case.patient_id not in ok or "seg" not in case.files:
            continue
        rows.append(patient_stats(case))
    if not rows:
        raise SystemExit("No labeled volumes could be read. No statistics were fabricated.")
    payload = aggregate_stats(rows)
    payload["inventory_summary"] = {
        "n_cases": inventory.n_cases,
        "n_complete": inventory.n_complete,
        "layout": inventory.layout,
        "naming_counts": inventory.naming_counts,
    }
    destination = args.out or (
        project_root() / cfg["paths"]["processed_root"] / "reports" / "stats.json"
    )
    write_json(payload, destination)
    print(f"patients={payload['n_patients']} class_voxels={payload['class_voxels']} -> {destination}")
    print("Research prototype. Not a clinically validated diagnostic tool.")


if __name__ == "__main__":
    main()
