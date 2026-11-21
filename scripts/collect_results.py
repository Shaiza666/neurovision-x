"""Build Phase 5 reports from recorded experiment JSON.

Does not train, download data, or fill missing Dice/HD95. Smoke folders are
inventoried and excluded from A/B/C tables.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from neurovisionx.experiments.catalog import catalog_experiments
from neurovisionx.experiments.plots import save_run_inventory, save_smoke_curves
from neurovisionx.experiments.report import (
    build_results_markdown,
    build_summary_markdown,
    load_evaluation_config,
)
from neurovisionx.utils.paths import project_root


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Write experimental_results.md from on-disk runs. Does not train."
    )
    parser.add_argument(
        "--experiments-root",
        type=Path,
        default=None,
        help="Override experiments/ directory.",
    )
    parser.add_argument(
        "--docs-root",
        type=Path,
        default=None,
        help="Where to write experimental_results.md (default: docs/).",
    )
    args = parser.parse_args()
    spec = load_evaluation_config()
    if not spec.get("expensive_training_requires_approval", True):
        raise SystemExit("evaluation.yaml must keep expensive_training_requires_approval: true.")
    runs = catalog_experiments(args.experiments_root)
    root = project_root()
    docs_root = args.docs_root or (root / "docs")
    fig_dir = docs_root / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    figure_rels: dict[str, str] = {}
    try:
        save_run_inventory(runs, fig_dir / "phase5_run_inventory.png")
        figure_rels["inventory"] = "figures/phase5_run_inventory.png"
    except ImportError:
        pass
    smokes = [row for row in runs if row.get("smoke") and row.get("history")]
    if smokes:
        try:
            save_smoke_curves(smokes[0]["history"], fig_dir / "smoke_not_research_curves.png")
            figure_rels["smoke"] = "figures/smoke_not_research_curves.png"
        except ImportError:
            pass

    results = build_results_markdown(runs, spec, figure_rels=figure_rels)
    summary = build_summary_markdown(runs, spec)
    results_path = docs_root / "experimental_results.md"
    summary_path = docs_root / "experiment_summary.md"
    results_path.write_text(results, encoding="utf-8")
    summary_path.write_text(summary, encoding="utf-8")

    snapshot = {
        "n_runs": len(runs),
        "n_smoke": sum(1 for row in runs if row.get("smoke")),
        "n_research_eligible": sum(1 for row in runs if row.get("research_eligible")),
        "run_names": [row["run_name"] for row in runs],
        "clinical_disclaimer": "Research prototype. Not a clinically validated diagnostic tool.",
        "note": "Snapshot of folder names only. Metrics live in experimental_results.md.",
    }
    print(json.dumps(snapshot, indent=2))
    print(f"Wrote {results_path}")
    print(f"Wrote {summary_path}")
    print(
        "No training was started by this script. "
        "Cells without a non-smoke metrics_test.json stay 'not measured'."
    )


if __name__ == "__main__":
    main()
