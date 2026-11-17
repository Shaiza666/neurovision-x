"""Write a patient-level split JSON from validated, complete cases."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from neurovisionx.data.errors import MissingBraTSDataError
from neurovisionx.data.inventory import discover_inventory
from neurovisionx.data.splits import save_split, split_from_cases
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

    report = validate_cases(inventory.cases, require_complete=True, load_voxels=True)
    ok_ids = set(report.ok_ids())
    usable = [case for case in inventory.cases if case.patient_id in ok_ids]
    split_cfg = cfg["splits"]
    split = split_from_cases(
        usable,
        train_ratio=float(split_cfg["train_ratio"]),
        val_ratio=float(split_cfg["val_ratio"]),
        test_ratio=float(split_cfg["test_ratio"]),
        seed=int(split_cfg["seed"]),
        honor_official_validation=bool(split_cfg.get("honor_official_validation", True)),
    )
    destination = args.out or (project_root() / cfg["paths"]["splits_root"] / "patients.json")
    save_split(split, destination)
    meta = destination.with_name(destination.stem + "_meta.json")
    meta.write_text(
        json.dumps(
            {
                "n_complete_ok": len(usable),
                "train": len(split.train),
                "val": len(split.val),
                "test": len(split.test),
                "seed": split.seed,
                "layout": inventory.layout,
                "clinical_disclaimer": (
                    "Research prototype. Not a clinically validated diagnostic tool."
                ),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"train={len(split.train)} val={len(split.val)} test={len(split.test)} -> {destination}")


if __name__ == "__main__":
    main()
