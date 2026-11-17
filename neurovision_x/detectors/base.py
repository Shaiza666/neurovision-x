"""Shared detection types."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Detection:
    label: str
    score: float
    box: tuple[int, int, int, int]

    def as_row(self) -> dict[str, float | str | int]:
        x1, y1, x2, y2 = self.box
        return {
            "label": self.label,
            "score": round(self.score, 3),
            "x1": x1,
            "y1": y1,
            "x2": x2,
            "y2": y2,
        }
