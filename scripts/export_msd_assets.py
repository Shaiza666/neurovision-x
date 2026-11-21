"""Copy real MSD CPU figures into docs/ and write the measured appendix.

Reads experiments/*/metrics_test.json and learning_curves.png only.
Does not train and does not invent scores. Run after evaluate.py.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from neurovisionx.evaluation.summarize import format_summary
from neurovisionx.utils.config import load_yaml


RUNS = ("A0_baseline", "A2_attention", "A5_combination")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _bar_chart(metrics: dict[str, dict], path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = ["Dice WT", "Dice TC", "Dice ET"]
    keys = ["dice_wt", "dice_tc", "dice_et"]
    names = []
    means = []
    for run in RUNS:
        summary = metrics.get(run, {}).get("summary") or {}
        if not summary:
            continue
        names.append(run)
        means.append([float((summary.get(key) or {}).get("mean") or 0.0) for key in keys])
    if not names:
        return
    import numpy as np

    x = np.arange(len(labels))
    width = 0.25
    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    for index, (name, row) in enumerate(zip(names, means)):
        ax.bar(x + (index - 1) * width, row, width, label=name)
    ax.set_xticks(x, labels)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Mean Dice (test)")
    ax.set_title("MSD Task01 reduced CPU protocol — not official BraTS, not clinical")
    ax.legend(fontsize=8)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=120)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiments", type=Path, default=ROOT / "experiments")
    parser.add_argument("--out-dir", type=Path, default=ROOT / "docs" / "figures" / "msd_cpu")
    parser.add_argument("--appendix", type=Path, default=ROOT / "docs" / "msd_measured_appendix.md")
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    loaded: dict[str, dict] = {}
    lines = [
        "## Measured MSD CPU run (generated from JSON)",
        "",
        "Figures and cells in this section were copied from `experiments/` after training. "
        "They are **MSD Task01**, reduced CPU protocol (`docs/msd_protocol.md`), "
        "**not** official BraTS 2021+ and **not** a clinical validation. "
        "Dice cells use `format_summary` (3 decimal places, sample std, Student-t interval). "
        "Unrounded values stay in `metrics_test.json`. HD95 is in **voxels** of the 2 mm grid.",
        "",
    ]
    for run in RUNS:
        folder = args.experiments / run
        metrics_path = folder / "metrics_test.json"
        if not metrics_path.is_file():
            lines.append(f"- `{run}`: metrics_test.json missing. Not measured.")
            continue
        payload = json.loads(metrics_path.read_text(encoding="utf-8"))
        loaded[run] = payload
        summary = payload.get("summary") or {}
        lines.append(f"### `{run}`")
        lines.append("")
        lines.append(f"- patients: {payload.get('n_patients')}")
        lines.append(f"- inference: `{json.dumps(payload.get('inference'))}`")
        lines.append(f"- spatial: `{json.dumps(payload.get('spatial'))}`")
        for key in ("dice_wt", "dice_tc", "dice_et", "hd95_wt", "hd95_tc", "hd95_et", "ece", "brier"):
            lines.append(f"- {key}: {format_summary(summary.get(key))}")
        curve = folder / "learning_curves.png"
        if curve.is_file():
            dest = args.out_dir / f"{run}_learning_curves.png"
            shutil.copyfile(curve, dest)
            lines.append(f"- learning curves: `figures/msd_cpu/{dest.name}`")
        ckpt = folder / "best.pt"
        if ckpt.is_file():
            lines.append(f"- best.pt sha256: `{_sha256(ckpt)}`")
            lines.append(f"- best.pt bytes: {ckpt.stat().st_size}")
        lines.append("")

    chart = args.out_dir / "dice_means.png"
    if loaded:
        _bar_chart(loaded, chart)
        lines.extend(
            [
                f"![Mean test Dice](figures/msd_cpu/{chart.name})",
                "",
                "*Mean test Dice on the reduced MSD cohort. Not a BraTS leaderboard chart.*",
                "",
            ]
        )
    # Qualitative panels copied if explain.py already wrote them.
    for run in RUNS:
        explain_root = args.experiments / run / "explain"
        if not explain_root.is_dir():
            continue
        overlays = sorted(explain_root.rglob("orthogonal_overlay.png"))
        attributions = sorted(explain_root.rglob("orthogonal_attribution.png"))
        if overlays:
            dest = args.out_dir / f"{run}_overlay.png"
            shutil.copyfile(overlays[0], dest)
            lines.append(f"![{run} overlay](figures/msd_cpu/{dest.name})")
            lines.append("")
            lines.append(f"*{run}: FLAIR with prediction overlay from the checkpoint forward. MSD cohort, not clinical.*")
            lines.append("")
        if attributions:
            dest = args.out_dir / f"{run}_gradcam.png"
            shutil.copyfile(attributions[0], dest)
            lines.append(f"![{run} Grad-CAM](figures/msd_cpu/{dest.name})")
            lines.append("")
    uncertainty = args.experiments / "A5_combination" / "qc_overlays"
    if uncertainty.is_dir():
        overlays = sorted(uncertainty.glob("*_overlay.png"))
        if overlays:
            dest = args.out_dir / "A5_uncertainty.png"
            shutil.copyfile(overlays[0], dest)
            lines.append(f"![A5 uncertainty](figures/msd_cpu/{dest.name})")
            lines.append("")
            lines.append("*A5 MC-dropout uncertainty overlay written by analyze_reliability.py. Not a clinical confidence map.*")
            lines.append("")
    rel = args.experiments / "A5_combination" / "reliability_diagram.png"
    if rel.is_file():
        dest = args.out_dir / "A5_reliability.png"
        shutil.copyfile(rel, dest)
        lines.append(f"![A5 reliability](figures/msd_cpu/{dest.name})")
        lines.append("")
    uncertainty_png = args.out_dir / "A5_uncertainty.png"
    if uncertainty_png.is_file():
        lines.append(f"![A5 uncertainty](figures/msd_cpu/{uncertainty_png.name})")
        lines.append("")
        lines.append(
            "*A5 MC-dropout entropy on an axial slice of `test/BRATS_422` (T=4 for this figure; "
            "test metrics use T=10). Not a clinical confidence map.*"
        )
        lines.append("")
    lines.extend(
        [
            "### Reading the numbers",
            "",
            "- HD95 is in voxels of the 2 mm grid. One voxel is 2 mm, so a millimetre figure is the voxel mean times 2. That product is derived, not a second measurement.",
            "- `test/BRATS_026` has no enhancing-tumor surface, so HD95 ET is NaN and is omitted (n=11). Dice ET for an empty reference is still included (n=12).",
            "- A Student-t interval can cross zero when the standard deviation is large and n is small (A2 HD95 ET). The distances themselves are not negative.",
            "- A5 Spearman ρ (patient entropy vs ET Dice) on this split is positive and the p-value is large. H2 is not supported here. A0/A2 entropy is T=1, not MC dropout.",
            "- Component ablations C−SE, C−boundary, and C−MC were not run. A vs B vs C is one seed (2026) on 12 test patients. No method is claimed to be better in general.",
            "",
        ]
    )

    args.appendix.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
    print(f"Wrote {args.appendix}")

    # Record digests. Leave url null; this script does not upload.
    ckpt_cfg = ROOT / "configs" / "checkpoints.yaml"
    payload = load_yaml(ckpt_cfg)
    for run in RUNS:
        ckpt = args.experiments / run / "best.pt"
        if run in (payload.get("runs") or {}) and ckpt.is_file():
            payload["runs"][run]["sha256"] = _sha256(ckpt)
            payload["runs"][run]["bytes"] = ckpt.stat().st_size
    # Preserve a readable YAML dump without pulling in extra formatting deps.
    import yaml

    ckpt_cfg.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    print(f"Updated {ckpt_cfg}")


if __name__ == "__main__":
    main()
