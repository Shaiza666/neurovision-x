"""Training entry points.

Training is reported complete only after a real job on official data, except
the explicit ``--smoke`` path which uses random tensors and is labeled as such.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from neurovisionx.data.dataset import MissingBraTSDataError, require_brats_root
from neurovisionx.utils.config import load_experiment_config
from neurovisionx.utils.metadata import build_run_metadata, write_metadata
from neurovisionx.utils.paths import project_root
from neurovisionx.utils.seed import seed_everything


@dataclass
class TrainRequest:
    data_root: Path
    run_dir: Path
    config: dict[str, Any]


def refuse_without_real_data(raw_root: Path) -> None:
    require_brats_root(raw_root)


def prepare_run(
    run_name: str = "debug",
    config: dict[str, Any] | None = None,
    extra_metadata: dict[str, Any] | None = None,
) -> TrainRequest:
    root = project_root()
    config = config if config is not None else load_experiment_config()
    seed_everything(int(config["training"]["seed"]), bool(config["training"].get("deterministic", True)))
    raw = root / config["data"]["paths"]["raw_root"]
    refuse_without_real_data(raw)
    run_dir = root / config["data"]["paths"]["output_root"] / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    meta = {"run_name": run_name}
    if extra_metadata:
        meta.update(extra_metadata)
    write_metadata(run_dir / "metadata.json", build_run_metadata(config, meta))
    return TrainRequest(data_root=raw, run_dir=run_dir, config=config)


def train_or_explain(
    run_name: str = "debug",
    *,
    smoke: bool = False,
    resume: bool = False,
    ablation_id: str | None = None,
    config: dict[str, Any] | None = None,
) -> None:
    extra_metadata: dict[str, Any] = {}
    if ablation_id:
        from neurovisionx.experiments.ablation import apply_ablation

        run, config = apply_ablation(ablation_id, config)
        extra_metadata["ablation_id"] = run.id
        extra_metadata["ablation_isolates"] = run.isolates
        if not run.train:
            raise SystemExit(
                f"Ablation {run.id} is evaluation-only (checkpoint_from="
                f"{run.checkpoint_from!r}). Do not retrain. After A0 finishes, "
                "run scripts/analyze_reliability.py --run-name A0_baseline."
            )
        if smoke:
            raise SystemExit("Do not combine --smoke with --ablation. Smoke is not BraTS.")
        if run_name == "debug":
            run_name = run.id
    if smoke:
        from neurovisionx.training.engine import run_smoke

        checkpoint = run_smoke(run_name)
        print(
            f"SMOKE TEST finished (synthetic tensors, not BraTS, not research results). "
            f"Checkpoint: {checkpoint}"
        )
        return
    try:
        request = prepare_run(run_name, config=config, extra_metadata=extra_metadata)
    except MissingBraTSDataError as exc:
        raise SystemExit(str(exc)) from exc
    from neurovisionx.training.engine import run_training

    checkpoint = run_training(request, resume=resume)
    print(f"Training finished. Checkpoint: {checkpoint}")
