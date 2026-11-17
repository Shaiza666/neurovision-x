"""Experiment metadata written next to checkpoints."""

from __future__ import annotations

import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from neurovisionx import __version__


def build_run_metadata(config: dict[str, Any], extra: dict[str, Any] | None = None) -> dict[str, Any]:
    payload = {
        "package_version": __version__,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "python": sys.version,
        "platform": platform.platform(),
        "config": config,
        "clinical_disclaimer": (
            "Research prototype. Not a clinically validated diagnostic tool."
        ),
    }
    if extra:
        payload.update(extra)
    try:
        import torch

        payload["torch"] = torch.__version__
        payload["cuda"] = torch.cuda.is_available()
    except ImportError:
        payload["torch"] = None
        payload["cuda"] = False
    return payload


def write_metadata(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
