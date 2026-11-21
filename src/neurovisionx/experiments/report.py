"""Markdown reports from the experiment catalog.

Comparison cells are 'not measured' unless a non-smoke test JSON exists.
This module never copies leaderboard numbers or smoke Dice into A/B/C tables.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from neurovisionx.evaluation.hard_cases import rank_failures
from neurovisionx.evaluation.summarize import format_summary, summarize_values
from neurovisionx.experiments.catalog import find_run
from neurovisionx.utils.config import load_yaml
from neurovisionx.utils.paths import project_root


NOT_MEASURED = "not measured"


def load_evaluation_config(path: Path | str | None = None) -> dict[str, Any]:
    resolved = Path(path) if path is not None else project_root() / "configs" / "evaluation.yaml"
    return load_yaml(resolved)


def _metric_from_run(run: dict[str, Any] | None, key: str) -> dict[str, Any] | None:
    if run is None or run.get("smoke") or not run.get("research_eligible"):
        return None
    metrics = run.get("metrics_test") or {}
    summary = metrics.get("summary") or {}
    if key in summary and isinstance(summary[key], dict) and summary[key].get("n"):
        return summary[key]
    patients = metrics.get("patients") or []
    values = [row[key] for row in patients if isinstance(row, dict) and key in row]
    if not values:
        rel = run.get("reliability_test") or {}
        rel_sum = rel.get("summary") or {}
        if key in rel_sum and isinstance(rel_sum[key], dict) and rel_sum[key].get("n"):
            return rel_sum[key]
        cal = rel.get("calibration") or {}
        if key in cal:
            item = cal[key]
            if isinstance(item, dict):
                return item
            if item is not None and item == item:
                return summarize_values([float(item)])
        if key == "ece" and "ece" in cal:
            ece = cal["ece"]
            if ece is not None:
                return summarize_values([float(ece)])
        return None
    return summarize_values(values)


def _cell(run: dict[str, Any] | None, key: str) -> str:
    return format_summary(_metric_from_run(run, key))


def _md_table(headers: list[str], rows: list[list[str]]) -> str:
    line = "| " + " | ".join(headers) + " |"
    sep = "| " + " | ".join("---" for _ in headers) + " |"
    body = ["| " + " | ".join(row) + " |" for row in rows]
    return "\n".join([line, sep, *body])


def comparison_table(
    runs: list[dict[str, Any]],
    spec: dict[str, Any],
    metric_keys: list[tuple[str, str]],
) -> str:
    headers = ["Metric"]
    arms = []
    for letter, arm in spec.get("comparisons", {}).items():
        headers.append(f"{letter}: {arm.get('label', letter)}")
        arms.append(find_run(runs, str(arm["run_id"])))
    rows = []
    for key, label in metric_keys:
        rows.append([label, *[_cell(arm, key) for arm in arms]])
    return _md_table(headers, rows)


def ablation_table(runs: list[dict[str, Any]], spec: dict[str, Any], metric: str) -> str:
    headers = ["Arm", "Removes", "Train?", f"{metric}"]
    rows = []
    for arm in spec.get("component_ablations", []):
        rec = find_run(runs, str(arm["run_id"]))
        rows.append(
            [
                str(arm["id"]),
                str(arm.get("removes", "")),
                "yes" if arm.get("train") else "no (reuse ckpt)",
                _cell(rec, metric),
            ]
        )
    return _md_table(headers, rows)


def inventory_table(runs: list[dict[str, Any]]) -> str:
    headers = ["Folder", "Status", "Smoke?", "Test metrics?", "Device", "CUDA", "Epochs"]
    rows = []
    for row in runs:
        rows.append(
            [
                str(row["run_name"]),
                str(row.get("status") or "unknown"),
                "yes" if row.get("smoke") else "no",
                "yes" if row.get("research_eligible") else "no",
                str(row.get("device") or "unknown"),
                str(row.get("cuda")),
                str(row.get("epochs_ran") if row.get("epochs_ran") is not None else "unknown"),
            ]
        )
    if not rows:
        rows.append(["(none)", "no recorded runs", "—", "—", "—", "—", "—"])
    return _md_table(headers, rows)


def failure_section(runs: list[dict[str, Any]], spec: dict[str, Any]) -> str:
    proposed = spec.get("comparisons", {}).get("C", {})
    rec = find_run(runs, str(proposed.get("run_id", "A5_combination")))
    if rec is None or rec.get("smoke") or not rec.get("research_eligible"):
        return (
            "No failure-case ranking is available. It requires a non-smoke "
            "`metrics_test.json` with per-patient Dice."
        )
    patients = (rec.get("metrics_test") or {}).get("patients") or []
    worst = rank_failures(patients, metric="dice_et", k=8, higher_is_better=True)
    if not worst:
        return "Test JSON exists but has no per-patient `dice_et` rows to rank."
    headers = ["patient_id", "dice_et", "hd95_et", "et_voxels"]
    rows = [
        [
            str(item.get("patient_id", "")),
            f"{item.get('dice_et'):.3f}" if item.get("dice_et") == item.get("dice_et") else "NaN",
            f"{item.get('hd95_et'):.3f}" if item.get("hd95_et") == item.get("hd95_et") else "NaN",
            str(item.get("et_voxels", "")),
        ]
        for item in worst
    ]
    return "Lowest ET Dice on the recorded test split (not a clinical list):\n\n" + _md_table(headers, rows)


def smoke_appendix(runs: list[dict[str, Any]]) -> str:
    smokes = [row for row in runs if row.get("smoke")]
    if not smokes:
        return "No synthetic smoke folder with `metadata.json` was found."
    lines = [
        "These numbers are **software verification only**. They used random tensors. "
        "They are **not** BraTS results and are **not** an A/B/C comparison.",
        "",
    ]
    for row in smokes:
        lines.append(f"### `{row['run_name']}`")
        lines.append("")
        lines.append(f"- status: `{row.get('status')}`")
        lines.append(f"- device: `{row.get('device')}`")
        lines.append(f"- CUDA available: `{row.get('cuda')}`")
        lines.append(f"- torch: `{row.get('torch')}`")
        lines.append(f"- epochs_ran: `{row.get('epochs_ran')}`")
        if row.get("n_params") is not None:
            lines.append(f"- n_params: `{row.get('n_params')}`")
        if row.get("note"):
            lines.append(f"- note: {row['note']}")
        history = row.get("history") or []
        if history:
            last = history[-1]
            lines.append(
                f"- last recorded synthetic val Dice WT/TC/ET: "
                f"{last.get('dice_wt')}, {last.get('dice_tc')}, {last.get('dice_et')} "
                "(random tensors; do not cite)"
            )
        lines.append("")
    return "\n".join(lines)


def _dataset_section(runs: list[dict[str, Any]]) -> list[str]:
    lines = ["## Dataset on disk for these runs", ""]
    seen = False
    for row in runs:
        if row.get("smoke"):
            continue
        data = ((row.get("metadata") or {}).get("config") or {}).get("data") or {}
        dataset = data.get("dataset") or {}
        if not dataset:
            continue
        seen = True
        lines.append(
            f"- `{row['run_name']}`: `{dataset.get('name', 'unknown')}` — {dataset.get('description', '').strip()}"
        )
        reductions = dataset.get("protocol_reductions")
        if reductions:
            lines.append(f"  - Protocol reductions: {str(reductions).strip()}")
        note = dataset.get("license_note")
        if note:
            lines.append(f"  - License: {str(note).strip()}")
    if not seen:
        lines.append(
            "No non-smoke run recorded a dataset block. Official BraTS volumes are not implied."
        )
    lines.append("")
    lines.append(
        "These rows are **not** an official BraTS / Synapse leaderboard comparison. "
        "MSD Task01 is BraTS 2016/2017-derived and CC-BY-SA 4.0. "
        "This software is not a clinically validated diagnostic tool."
    )
    lines.append("")
    return lines


def build_results_markdown(
    runs: list[dict[str, Any]],
    spec: dict[str, Any],
    *,
    figure_rels: dict[str, str] | None = None,
) -> str:
    figure_rels = figure_rels or {}
    now = datetime.now(timezone.utc).isoformat()
    n_research = sum(1 for row in runs if row.get("research_eligible"))
    n_smoke = sum(1 for row in runs if row.get("smoke"))
    metric_keys = [
        ("dice_wt", "Dice WT"),
        ("dice_tc", "Dice TC"),
        ("dice_et", "Dice ET"),
        ("hd95_wt", "HD95 WT"),
        ("hd95_tc", "HD95 TC"),
        ("hd95_et", "HD95 ET"),
        ("precision_et", "Precision ET"),
        ("recall_et", "Recall ET"),
        ("ece", "ECE"),
        ("brier", "Brier (multiclass)"),
    ]
    lines = [
        "# Experimental results (Phase 5)",
        "",
        "**This software is a research prototype, not a clinically validated diagnostic tool.**",
        "",
        f"Generated: `{now}` from on-disk `experiments/` JSON only. "
        "Missing measurements are written as **not measured**. "
        "No leaderboard number, no paper table, and no synthetic smoke Dice was copied into the comparison tables.",
        "",
        "## Verdict on recorded evidence",
        "",
        f"- Research-eligible test evaluations (non-smoke `metrics_test.json`): **{n_research}**",
        f"- Synthetic smoke folders: **{n_smoke}**",
        "- Official BraTS / Synapse A vs B vs C comparison: **not measured**.",
        (
            "- No research-eligible test JSON is on disk."
            if n_research == 0
            else f"- Research-eligible runs present: {n_research}. "
            "Numbers below are from those files only. They are not a leaderboard entry."
        ),
        "- No improvement of the proposed method over the baseline is claimed.",
        "",
        *_dataset_section(runs),
        "## Actual recorded runs",
        "",
        inventory_table(runs),
        "",
    ]
    if figure_rels.get("inventory"):
        lines.extend(
            [
                f"![Run inventory]({figure_rels['inventory']})",
                "",
                "*Figure 1. Count of folders that actually exist. Not a performance chart.*",
                "",
            ]
        )
    lines.extend(
        [
            "## Comparison (A / B / C)",
            "",
            "Cells stay empty until the named run has a non-smoke `metrics_test.json`. "
            "When a run's dataset block names a reduced protocol, these cells are that "
            "protocol, not the 1 mm BraTS schedule in `docs/evaluation_protocol.md`.",
            "",
            comparison_table(runs, spec, metric_keys),
            "",
            "### Additional planned metrics",
            "",
            "| Metric | Status |",
            "| --- | --- |",
            f"| Small-ET Dice (lowest quartile of ET voxels) | {NOT_MEASURED if n_research == 0 else 'see reliability JSON'} |",
            f"| Spearman ρ (patient entropy vs ET Dice) | {NOT_MEASURED if n_research == 0 else 'see reliability JSON'} |",
            f"| AUPRC for worst ET Dice quartile | {NOT_MEASURED if n_research == 0 else 'see reliability JSON'} |",
            f"| Inference latency (ms, T=1 and T=10) | {NOT_MEASURED if n_research == 0 else 'see reliability JSON'} |",
            f"| GPU peak allocated MB | {NOT_MEASURED if n_research == 0 else 'see reliability JSON'} |",
            "",
            "## Component ablations (from C)",
            "",
            "Each arm removes one proposed component. None of these jobs has been started "
            "unless a matching non-smoke folder exists.",
            "",
            ablation_table(runs, spec, "dice_et"),
            "",
            "## Failure cases",
            "",
            failure_section(runs, spec),
            "",
            "## Multi-seed protocol",
            "",
            "Configured seeds: "
            + ", ".join(str(s) for s in spec.get("seeds", []))
            + ". Additional seeds are **planned** and have not been run. "
            "Mean ± std and t-intervals across seeds will be filled only from completed seed folders.",
            "",
            "## Limitations",
            "",
            *(
                [
                    "- No research-eligible test JSON is on disk, so Dice/HD95/ECE are not measured.",
                    "- Official BraTS / Synapse volumes are not in this tree.",
                ]
                if n_research == 0
                else [
                    "- Measured Dice/HD95/ECE come only from non-smoke `metrics_test.json`.",
                    "- They are not official BraTS 2021+ leaderboard scores and not a clinical validation.",
                    "- HD95 in this codebase is in voxels. On a resampled grid, do not relabel it as millimetres.",
                ]
            ),
            "- The 50-epoch 1 mm BraTS protocol in `configs/training.yaml` is separate. "
            "`expensive_training_requires_approval` stays true for that protocol.",
            "- Smoke Dice is patch-wise on random cubes; it is not comparable to challenge scores.",
            "- GPU memory is undefined on CPU-only machines (`cuda: false`).",
            "- HD95 is NaN when a surface cannot be formed; NaNs are omitted from means, not replaced.",
            "- This is not a clinical validation.",
            "",
            "## Reproducibility",
            "",
            "Regenerate this file (still without inventing metrics) with:",
            "",
            "```powershell",
            ".\\.venv\\Scripts\\python scripts\\collect_results.py",
            "```",
            "",
            "After approved BraTS jobs:",
            "",
            "```powershell",
            ".\\.venv\\Scripts\\python scripts\\train.py --ablation A0_baseline --run-name A0_baseline",
            ".\\.venv\\Scripts\\python scripts\\evaluate.py --run-name A0_baseline",
            ".\\.venv\\Scripts\\python scripts\\analyze_reliability.py --run-name A0_baseline --samples 10",
            ".\\.venv\\Scripts\\python scripts\\collect_results.py",
            "```",
            "",
            "## Appendix: synthetic smoke (not research)",
            "",
            smoke_appendix(runs),
        ]
    )
    if figure_rels.get("smoke"):
        lines.extend(
            [
                "",
                f"![Smoke curves, not research]({figure_rels['smoke']})",
                "",
                "*Figure A1. Synthetic smoke learning curves. Do not cite as BraTS performance.*",
                "",
            ]
        )
    appendix = project_root() / "docs" / "msd_measured_appendix.md"
    if appendix.is_file():
        lines.extend(["", appendix.read_text(encoding="utf-8").rstrip(), ""])
    return "\n".join(lines).rstrip() + "\n"


def build_summary_markdown(runs: list[dict[str, Any]], spec: dict[str, Any]) -> str:
    n_research = sum(1 for row in runs if row.get("research_eligible"))
    n_smoke = sum(1 for row in runs if row.get("smoke"))
    now = datetime.now(timezone.utc).isoformat()
    return "\n".join(
        [
            "# Experiment summary (Phase 5)",
            "",
            "**Research prototype. Not a clinically validated diagnostic tool.**",
            "",
            f"Generated: `{now}`",
            "",
            "## What was actually measured",
            "",
            f"- {n_research} research-eligible test evaluation(s) from whatever volumes were on disk.",
            f"- {n_smoke} synthetic smoke run(s), excluded from A/B/C tables.",
            "- A (3D U-Net), B (Attention U-Net), and C (proposed SE + ET boundary + MC dropout) "
            "are **not compared on official data** (Synapse BraTS).",
            (
                "- No research-eligible test JSON is on disk."
                if n_research == 0
                else "- See `docs/experimental_results.md` for the recorded rows. "
                "They are not a BraTS leaderboard entry."
            ),
            "- Component ablations C−SE, C−boundary, C−MC stay **not measured** unless a matching non-smoke folder exists.",
            "- Multi-seed means: **not measured** (configured seeds: "
            + ", ".join(str(s) for s in spec.get("seeds", []))
            + ").",
            "",
            "## What is planned, not run",
            "",
            "Patient-level evaluation of A/B/C and the three component ablations, with Dice, HD95, "
            "precision/recall, small-ET stratum, ECE, Brier, uncertainty–error Spearman/AUPRC, "
            "latency, and GPU memory, after protocol approval and licensed BraTS volumes.",
            "",
            "## Claims that are not made",
            "",
            "The proposed method is **not** reported as more accurate, better calibrated, or faster. "
            "Those statements require completed non-smoke `metrics_test.json` files.",
            "",
        ]
    )
