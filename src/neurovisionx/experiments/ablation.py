"""Ablation plan loader. Does not start training.

Each arm isolates one Phase 4 component. Combining them (A5) is an extra
arm, not a novelty claim. Expensive BraTS jobs require protocol approval.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from neurovisionx.utils.config import load_experiment_config, load_yaml
from neurovisionx.utils.paths import project_root


@dataclass(frozen=True)
class AblationRun:
    id: str
    train: bool
    description: str
    isolates: str
    checkpoint_from: str | None
    spec: dict[str, Any]

    def apply_to(self, config: dict[str, Any]) -> dict[str, Any]:
        return apply_overrides(config, self.spec)


def _merge(dst: dict[str, Any], src: dict[str, Any]) -> dict[str, Any]:
    for key, value in src.items():
        if isinstance(value, dict) and isinstance(dst.get(key), dict):
            _merge(dst[key], value)
        else:
            dst[key] = deepcopy(value)
    return dst


def apply_overrides(config: dict[str, Any], spec: dict[str, Any]) -> dict[str, Any]:
    merged = deepcopy(config)
    if "model" in spec:
        _merge(merged.setdefault("model", {}), spec["model"])
    if "loss" in spec:
        _merge(merged.setdefault("training", {}).setdefault("loss", {}), spec["loss"])
    if "uncertainty" in spec:
        _merge(merged.setdefault("training", {}).setdefault("uncertainty", {}), spec["uncertainty"])
    if "training" in spec:
        _merge(merged.setdefault("training", {}), spec["training"])
    return merged


def load_ablation_plan(path: Path | str | None = None) -> dict[str, Any]:
    resolved = Path(path) if path is not None else project_root() / "configs" / "ablations.yaml"
    plan = load_yaml(resolved)
    if not isinstance(plan, dict) or "runs" not in plan:
        raise ValueError(f"Expected an ablation plan with a 'runs' list in {resolved}")
    return plan


def list_ablation_runs(plan: dict[str, Any] | None = None) -> list[AblationRun]:
    payload = plan if plan is not None else load_ablation_plan()
    runs: list[AblationRun] = []
    for raw in payload["runs"]:
        runs.append(
            AblationRun(
                id=str(raw["id"]),
                train=bool(raw.get("train", True)),
                description=str(raw.get("description", "")),
                isolates=str(raw.get("isolates", "")),
                checkpoint_from=raw.get("checkpoint_from"),
                spec=raw,
            )
        )
    return runs


def apply_ablation(
    ablation_id: str,
    config: dict[str, Any] | None = None,
    plan: dict[str, Any] | None = None,
) -> tuple[AblationRun, dict[str, Any]]:
    """Return the named arm and a full experiment config with its overrides."""
    payload = plan if plan is not None else load_ablation_plan()
    runs = {run.id: run for run in list_ablation_runs(payload)}
    if ablation_id not in runs:
        known = ", ".join(sorted(runs))
        raise KeyError(f"Unknown ablation {ablation_id!r}. Known: {known}")
    run = runs[ablation_id]
    base = deepcopy(config) if config is not None else load_experiment_config()
    return run, run.apply_to(base)


def format_plan_table(runs: list[AblationRun] | None = None) -> str:
    rows = runs if runs is not None else list_ablation_runs()
    header = (
        f"{'ID':<16} {'TRAIN':<6} {'ISOLATES':<22} {'DESCRIPTION'}\n"
        f"{'-' * 16} {'-' * 6} {'-' * 22} {'-' * 40}"
    )
    lines = [
        "NeuroVision-X Phase 4 ablation plan (no training started).",
        "Approve docs/experimental_protocol.md before any expensive BraTS job.",
        "Research prototype — not a clinically validated diagnostic tool.",
        "",
        header,
    ]
    for run in rows:
        lines.append(
            f"{run.id:<16} {'yes' if run.train else 'no':<6} {run.isolates:<22} {run.description}"
        )
    lines.append("")
    lines.append("A4 reuses A0 weights (MC dropout at test). Do not retrain A4.")
    return "\n".join(lines)
