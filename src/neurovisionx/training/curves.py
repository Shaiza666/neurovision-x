"""Learning-curve figures from logged history.

Curves are research plots. They are not clinical performance charts.
Synthetic smoke runs must be titled as such.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any


def save_learning_curves(
    history: list[dict[str, float]],
    destination: Path,
    *,
    title: str,
) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    destination.parent.mkdir(parents=True, exist_ok=True)
    epochs = [row.get("epoch", i + 1) for i, row in enumerate(history)]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(epochs, [row.get("train_loss", float("nan")) for row in history], label="train")
    axes[0].plot(epochs, [row.get("val_loss", float("nan")) for row in history], label="val")
    axes[0].set_xlabel("epoch")
    axes[0].set_ylabel("loss")
    axes[0].legend()
    axes[0].set_title("Loss")
    for key, label in (("dice_wt", "WT"), ("dice_tc", "TC"), ("dice_et", "ET")):
        axes[1].plot(epochs, [row.get(key, float("nan")) for row in history], label=label)
    axes[1].set_xlabel("epoch")
    axes[1].set_ylabel("Dice")
    axes[1].legend()
    axes[1].set_title("Nested Dice (val patches)")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(destination, dpi=120)
    plt.close(fig)
    return destination


def history_to_jsonable(history: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for row in history:
        converted = {}
        for key, value in row.items():
            if isinstance(value, float) and value != value:  # NaN
                converted[key] = None
            else:
                converted[key] = value
        out.append(converted)
    return out
