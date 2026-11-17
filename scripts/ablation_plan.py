"""Print the Phase 4 ablation plan. Does not train and does not load BraTS."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from neurovisionx.experiments.ablation import format_plan_table, list_ablation_runs, load_ablation_plan


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Print the Phase 4 ablation table. Does not start training."
    )
    parser.add_argument("--json", action="store_true", help="Dump the YAML plan as JSON")
    args = parser.parse_args()
    plan = load_ablation_plan()
    if not plan.get("expensive_training_requires_approval", True):
        raise SystemExit("Ablation plan must keep expensive_training_requires_approval: true.")
    if args.json:
        print(json.dumps(plan, indent=2))
        return
    print(format_plan_table(list_ablation_runs(plan)))
    print()
    print("This script did not launch training, evaluation, or data downloads.")


if __name__ == "__main__":
    main()
