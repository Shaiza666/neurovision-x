"""Compare two on-disk metrics JSON files for H1-style paired tests.

Does not train. Does not invent patient scores. Missing files exit honestly.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from neurovisionx.evaluation.hard_cases import paired_cohens_d, paired_wilcoxon
from neurovisionx.evaluation.tables import load_metrics_json
from neurovisionx.utils.paths import project_root


def _patient_metric(payload: dict, metric: str) -> dict[str, float]:
    rows = payload.get("patients") or []
    out: dict[str, float] = {}
    for row in rows:
        pid = row.get("patient_id")
        if pid is None or metric not in row:
            continue
        out[str(pid)] = float(row[metric])
    return out


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Paired comparison of two evaluate.py JSON files. Not a paper table."
    )
    parser.add_argument("--baseline", type=Path, required=True, help="metrics_test.json from A0")
    parser.add_argument("--proposed", type=Path, required=True, help="metrics_test.json from A3")
    parser.add_argument("--metric", default="hd95_et")
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Optional JSON path. Defaults to stdout only if omitted.",
    )
    args = parser.parse_args()
    root = project_root()
    baseline_path = args.baseline if args.baseline.is_absolute() else root / args.baseline
    proposed_path = args.proposed if args.proposed.is_absolute() else root / args.proposed
    if not baseline_path.is_file() or not proposed_path.is_file():
        raise SystemExit(
            "Both --baseline and --proposed must be existing metrics JSON from "
            "scripts/evaluate.py on official BraTS patients. No comparison was invented."
        )

    base = load_metrics_json(baseline_path)
    prop = load_metrics_json(proposed_path)
    if base.get("smoke") or prop.get("smoke"):
        raise SystemExit("Smoke metrics cannot enter an H1 comparison.")
    a_map = _patient_metric(base, args.metric)
    b_map = _patient_metric(prop, args.metric)
    shared = sorted(set(a_map) & set(b_map))
    if not shared:
        raise SystemExit(
            f"No shared patient_id values for {args.metric}. Comparison was not run."
        )
    import numpy as np

    a = np.array([a_map[pid] for pid in shared], dtype=np.float64)
    b = np.array([b_map[pid] for pid in shared], dtype=np.float64)
    report = {
        "metric": args.metric,
        "n_paired": len(shared),
        "baseline": str(baseline_path),
        "proposed": str(proposed_path),
        "wilcoxon": paired_wilcoxon(a, b),
        "cohens_d_paired": paired_cohens_d(a, b),
        "hypothesis": "H1 uses ET HD95 (lower proposed better) and ET Dice non-inferiority separately.",
        "supported": False,
        "note": (
            "This script only computes statistics from the JSON you pass. "
            "It does not claim H1 is supported. Research prototype — not a clinical report."
        ),
        "clinical_disclaimer": "Research prototype. Not a clinically validated diagnostic tool.",
    }
    text = json.dumps(report, indent=2)
    print(text)
    if args.out:
        out = args.out if args.out.is_absolute() else root / args.out
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
