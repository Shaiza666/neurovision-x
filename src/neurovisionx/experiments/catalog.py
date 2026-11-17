"""Inventory of recorded experiment directories.

Only JSON that exists on disk is trusted. Smoke runs are tagged and must not
enter A/B/C comparison tables. Missing metrics stay missing.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from neurovisionx.utils.paths import project_root

SMOKE_STATUSES = {"smoke_synthetic_not_research"}


def _read_json(path: Path) -> dict[str, Any] | list | None:
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return payload


def experiments_root(config: dict[str, Any] | None = None) -> Path:
    if config and "data" in config:
        return project_root() / config["data"]["paths"]["output_root"]
    return project_root() / "experiments"


def catalog_run(run_dir: Path) -> dict[str, Any]:
    """Describe one run folder. Does not compute Dice."""
    meta = _read_json(run_dir / "metadata.json")
    if not isinstance(meta, dict):
        meta = {}
    status = str(meta.get("status") or "unknown")
    smoke = bool(meta.get("smoke")) or status in SMOKE_STATUSES
    metrics_test = _read_json(run_dir / "metrics_test.json")
    reliability = _read_json(run_dir / "reliability_test.json")
    history = _read_json(run_dir / "history.json")
    has_ckpt = (run_dir / "best.pt").is_file() or (run_dir / "latest.pt").is_file()
    research_metrics = (
        not smoke
        and isinstance(metrics_test, dict)
        and int(metrics_test.get("n_patients") or 0) > 0
    )
    return {
        "run_name": run_dir.name,
        "path": str(run_dir),
        "status": status,
        "smoke": smoke,
        "research_eligible": research_metrics,
        "has_checkpoint": has_ckpt,
        "has_metrics_test": isinstance(metrics_test, dict),
        "has_reliability_test": isinstance(reliability, dict),
        "n_history_epochs": len(history) if isinstance(history, list) else 0,
        "device": meta.get("device"),
        "cuda": meta.get("cuda"),
        "torch": meta.get("torch"),
        "n_params": meta.get("n_params"),
        "epochs_ran": meta.get("epochs_ran"),
        "note": meta.get("note"),
        "metadata": meta,
        "metrics_test": metrics_test if isinstance(metrics_test, dict) else None,
        "reliability_test": reliability if isinstance(reliability, dict) else None,
        "history": history if isinstance(history, list) else None,
    }


def catalog_experiments(root: Path | None = None) -> list[dict[str, Any]]:
    base = root if root is not None else experiments_root()
    if not base.is_dir():
        return []
    runs = []
    for child in sorted(base.iterdir()):
        if not child.is_dir():
            continue
        if child.name in {"results", "qc"}:
            continue
        if not (child / "metadata.json").is_file():
            continue
        runs.append(catalog_run(child))
    return runs


def find_run(runs: list[dict[str, Any]], run_id: str) -> dict[str, Any] | None:
    for row in runs:
        if row["run_name"] == run_id or row["run_name"].startswith(f"{run_id}_seed"):
            return row
    return None
