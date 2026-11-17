"""Validate discovered BraTS cases. Writes a JSON report; invents no volumes."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from neurovisionx.data.errors import MissingBraTSDataError
from neurovisionx.data.inventory import discover_inventory
from neurovisionx.data.validate import validate_cases
from neurovisionx.utils.config import load_yaml
from neurovisionx.utils.paths import project_root


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate BraTS NIfTI files found on disk.")
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

    prep = cfg.get("preprocessing", {})
    spacing = tuple(float(v) for v in prep.get("target_spacing_mm", [1.0, 1.0, 1.0]))
    report = validate_cases(
        inventory.cases,
        spacing_atol_mm=float(cfg.get("validation", {}).get("spacing_atol_mm", 0.15)),
        target_spacing_mm=spacing,
        require_complete=True,
        load_voxels=True,
    )
    destination = args.out or (
        project_root() / cfg["paths"]["processed_root"] / "reports" / "validation.json"
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "inventory": inventory.to_dict(),
        "validation": report.to_dict(),
    }
    destination.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(
        f"cases={inventory.n_cases} complete={inventory.n_complete} "
        f"ok={report.n_ok} failed={report.n_failed} -> {destination}"
    )
    print("Research prototype. Not a clinically validated diagnostic tool.")
    if report.n_ok == 0:
        raise SystemExit("No patient passed validation.")


if __name__ == "__main__":
    main()
