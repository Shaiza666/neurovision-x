# Experiments

Each run should create a subdirectory with `metadata.json`, configs, checkpoints, and metrics logged from an actual training job. Empty folders mean no training has been executed.

Phase 5: `scripts/collect_results.py` inventories these folders and writes `docs/experimental_results.md`. Smoke jobs (`status: smoke_synthetic_not_research`) are never copied into A/B/C comparison tables. Do not place hand-edited Dice scores here.
