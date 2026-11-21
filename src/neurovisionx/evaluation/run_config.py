"""Pick the experiment config that matches a saved run.

Evaluation must follow the checkpoint's data root, patch size, and
uncertainty settings when those were stored at training time. Falling back
to the default BraTS YAML is only for runs that did not record a config.
"""

from __future__ import annotations

from typing import Any


def config_from_checkpoint(payload: dict[str, Any] | None, fallback: dict[str, Any]) -> dict[str, Any]:
    if isinstance(payload, dict):
        stored = payload.get("config")
        if (
            isinstance(stored, dict)
            and isinstance(stored.get("data"), dict)
            and isinstance(stored.get("training"), dict)
            and isinstance(stored.get("model"), dict)
        ):
            return stored
    return fallback
