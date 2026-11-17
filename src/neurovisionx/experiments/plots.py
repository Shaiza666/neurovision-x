"""Publication-style figures from recorded JSON only.

No figure is filled with placeholder Dice. Smoke curves are watermarked
'NOT RESEARCH'. Empty research comparisons yield an inventory plot only.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np


def _plt():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "figure.dpi": 300,
            "savefig.dpi": 300,
            "font.size": 10,
            "axes.labelsize": 11,
            "axes.titlesize": 11,
            "axes.grid": True,
            "grid.alpha": 0.3,
            "savefig.bbox": "tight",
        }
    )
    return plt


def save_run_inventory(runs: list[dict[str, Any]], path: Path) -> Path:
    plt = _plt()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    n_smoke = sum(1 for row in runs if row.get("smoke"))
    n_research = sum(1 for row in runs if row.get("research_eligible"))
    n_incomplete = sum(
        1 for row in runs if not row.get("smoke") and not row.get("research_eligible")
    )
    fig, ax = plt.subplots(figsize=(5.2, 3.4))
    labels = ["Smoke\n(not research)", "Incomplete\n(no test JSON)", "Research\n(test metrics)"]
    counts = [n_smoke, n_incomplete, n_research]
    bars = ax.bar(labels, counts, color=["#bdbdbd", "#f9a825", "#1565c0"])
    ax.set_ylabel("Recorded run folders")
    ax.set_title("Phase 5 run inventory (on-disk only)")
    ymax = max(counts + [1])
    ax.set_ylim(0, ymax + 0.75)
    for bar, count in zip(bars, counts):
        ax.text(bar.get_x() + bar.get_width() / 2, count + 0.05, str(count), ha="center")
    fig.savefig(path)
    plt.close(fig)
    return path


def save_smoke_curves(history: list[dict[str, Any]], path: Path) -> Path:
    plt = _plt()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    epochs = [row.get("epoch", i + 1) for i, row in enumerate(history)]
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.4))
    axes[0].plot(epochs, [row.get("train_loss") for row in history], label="train")
    axes[0].plot(epochs, [row.get("val_loss") for row in history], label="val")
    axes[0].set_xlabel("epoch")
    axes[0].set_ylabel("loss")
    axes[0].legend()
    for key, label in (("dice_wt", "WT"), ("dice_tc", "TC"), ("dice_et", "ET")):
        axes[1].plot(epochs, [row.get(key) for row in history], label=label)
    axes[1].set_xlabel("epoch")
    axes[1].set_ylabel("Dice (synthetic patches)")
    axes[1].legend()
    fig.suptitle("SMOKE synthetic tensors — NOT BraTS — NOT research results")
    fig.savefig(path)
    plt.close(fig)
    return path


def save_grouped_metric(
    series: dict[str, dict[str, Any]],
    metric: str,
    path: Path,
    ylabel: str,
    title: str,
) -> Path | None:
    """Error-bar plot. Returns None if every arm is 'not measured'."""
    labels = []
    means = []
    yerr = []
    any_measured = False
    for name, summary in series.items():
        row = (summary or {}).get(metric) or {}
        labels.append(name)
        if row.get("mean") is None:
            means.append(np.nan)
            yerr.append(0.0)
        else:
            any_measured = True
            means.append(float(row["mean"]))
            yerr.append(float(row["std"] or 0.0))
    if not any_measured:
        return None
    plt = _plt()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6.2, 3.6))
    x = np.arange(len(labels))
    ax.bar(x, means, yerr=yerr, capsize=4, color="#1565c0")
    ax.set_xticks(x, labels, rotation=15, ha="right")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    fig.savefig(path)
    plt.close(fig)
    return path
