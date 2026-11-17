"""YAML experiment configuration."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from neurovisionx.utils.paths import project_root


def load_yaml(path: Path | str) -> dict[str, Any]:
    resolved = Path(path)
    if not resolved.is_absolute():
        resolved = project_root() / resolved
    with resolved.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"Expected a mapping in {resolved}")
    return data


def load_experiment_config(
    data_path: Path | str = "configs/data.yaml",
    model_path: Path | str = "configs/model.yaml",
    training_path: Path | str = "configs/training.yaml",
) -> dict[str, Any]:
    """Load the three config files into one nested dict."""
    return {
        "data": load_yaml(data_path),
        "model": load_yaml(model_path),
        "training": load_yaml(training_path),
    }
