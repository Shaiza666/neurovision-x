"""Write QC slice figures for a sample of validated patients."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from neurovisionx.data.errors import MissingBraTSDataError
from neurovisionx.data.inventory import discover_inventory
from neurovisionx.data.preprocess import preprocess_patient_files, stack_modalities
from neurovisionx.data.qc import save_class_histogram, save_patient_qc
from neurovisionx.data.stats import aggregate_stats, patient_stats
from neurovisionx.data.validate import validate_cases
from neurovisionx.utils.config import load_yaml
from neurovisionx.utils.paths import project_root


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/data.yaml")
    parser.add_argument("--raw-root", type=Path, default=None)
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--n-patients", type=int, default=None)
    args = parser.parse_args()
    cfg = load_yaml(args.config)
    raw = Path(args.raw_root) if args.raw_root else project_root() / cfg["paths"]["raw_root"]
    try:
        inventory = discover_inventory(raw, max_depth=int(cfg.get("discovery", {}).get("max_depth", 6)))
    except MissingBraTSDataError as exc:
        raise SystemExit(str(exc)) from exc

    report = validate_cases(inventory.cases, require_complete=True, load_voxels=True)
    ok_cases = [case for case in inventory.cases if case.patient_id in set(report.ok_ids())]
    if not ok_cases:
        raise SystemExit("No validated patients. QC figures were not written.")

    n_show = int(args.n_patients or cfg.get("qc", {}).get("n_patients", 8))
    out_dir = args.out_dir or (project_root() / cfg["paths"]["processed_root"] / "qc")
    spacing = tuple(float(v) for v in cfg["preprocessing"]["target_spacing_mm"])
    crop_fg = bool(cfg["preprocessing"].get("crop_foreground", True))

    for case in ok_cases[:n_show]:
        arrays = preprocess_patient_files(
            case.path_map(),
            target_spacing_mm=spacing,
            crop_foreground=crop_fg,
        )
        image = stack_modalities(arrays)
        safe_name = case.patient_id.replace("/", "_").replace("\\", "_")
        save_patient_qc(
            image,
            arrays["seg"],
            out_dir / f"{safe_name}.png",
            title=case.patient_id,
        )

    labeled = [patient_stats(case) for case in ok_cases if "seg" in case.files]
    if labeled:
        payload = aggregate_stats(labeled)
        save_class_histogram(payload["class_voxels"], out_dir / "class_histogram.png")
        save_class_histogram(payload["region_voxels"], out_dir / "region_histogram.png")
    print(f"Wrote QC for {min(n_show, len(ok_cases))} patients -> {out_dir}")
    print("Research QC only. Not a clinical image set.")


if __name__ == "__main__":
    main()
