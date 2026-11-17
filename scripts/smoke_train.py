"""Synthetic-data smoke test for tensor shapes and training logic.

Never treat the printed losses or Dice as BraTS or publication numbers.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from neurovisionx.training.loop import train_or_explain


if __name__ == "__main__":
    train_or_explain(run_name="smoke", smoke=True)
