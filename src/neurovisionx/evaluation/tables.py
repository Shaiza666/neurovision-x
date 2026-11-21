"""Write patient-level evaluation tables. Values must already be computed."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any


def write_patient_csv(rows: list[dict[str, Any]], path: Path | str) -> Path:
    """Write one CSV row per patient. Does not invent missing metrics."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        out.write_text("patient_id\n", encoding="utf-8")
        return out
    keys: list[str] = []
    seen: set[str] = set()
    preferred = ("patient_id",)
    for key in preferred:
        if any(key in row for row in rows):
            keys.append(key)
            seen.add(key)
    for row in rows:
        for key in row:
            if key not in seen:
                keys.append(key)
                seen.add(key)
    with out.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in keys})
    return out


def scores_to_csv_text(scores: dict[str, Any]) -> str:
    """Single-volume scores as two-column CSV text for dashboard export."""
    lines = ["metric,value"]
    for key, value in scores.items():
        lines.append(f"{key},{value}")
    return "\n".join(lines) + "\n"


def load_metrics_json(path: Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected object in {path}")
    return payload
