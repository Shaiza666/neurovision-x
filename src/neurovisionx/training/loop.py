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


def prepare_run(run_name: str = "debug") -> TrainRequest:
    root = project_root()
    config = load_experiment_config()
    seed_everything(int(config["training"]["seed"]), bool(config["training"].get("deterministic", True)))
    raw = root / config["data"]["paths"]["raw_root"]
    refuse_without_real_data(raw)
    run_dir = root / config["data"]["paths"]["output_root"] / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    write_metadata(run_dir / "metadata.json", build_run_metadata(config, {"run_name": run_name}))
    return TrainRequest(data_root=raw, run_dir=run_dir, config=config)


def train_or_explain(run_name: str = "debug", *, smoke: bool = False, resume: bool = False) -> None:
    if smoke:
        from neurovisionx.training.engine import run_smoke

        checkpoint = run_smoke(run_name)
        print(
            f"SMOKE TEST finished (synthetic tensors, not BraTS, not research results). "
            f"Checkpoint: {checkpoint}"
        )
        return
    try:
        request = prepare_run(run_name)
    except MissingBraTSDataError as exc:
        raise SystemExit(str(exc)) from exc
    from neurovisionx.training.engine import run_training

    checkpoint = run_training(request, resume=resume)
    print(f"Training finished. Checkpoint: {checkpoint}")
