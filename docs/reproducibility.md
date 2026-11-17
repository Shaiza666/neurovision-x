# Reproducibility

This software is a research prototype, not a clinically validated diagnostic tool.

## What is recorded

Each training run under `experiments/<run-name>/` should contain:

- `metadata.json` — package version, Python, platform, CUDA flag, full YAML config
- `patients.json` — patient-level train/val/test ids (no overlapping patients)
- `history.json` — per-epoch losses and nested val Dice from the job that actually ran
- `learning_curves.png` — research plot, not a clinical chart
- `best.pt` / `latest.pt` — best monitored metric and last epoch (optimizer included for `--resume`)

A run with `metadata.json` `"status": "smoke_synthetic_not_research"` used random tensors. Do not cite it as BraTS performance.

If those files are missing, training has not been completed. Do not copy metrics from papers into these files.

## Leakage controls

- Splits are **patient-level** (`configs/data.yaml`). Slices from one patient cannot appear in two splits.
- Intensity z-scoring uses non-zero voxels of the volume being loaded. Dataset-wide statistics, if added later, must be fit on **training patients only**.
- Augmentation and random crops are training-only (`random_crop=True`).

## Seeds

`configs/training.yaml` sets `seed: 2026` and `deterministic: true`. `seed_everything` sets Python, NumPy, and PyTorch seeds when those libraries are installed.

## Data

Official BraTS / Synapse volumes are required. `scripts/validate_dataset.py`, `scripts/prepare_splits.py`, `scripts/dataset_stats.py`, `scripts/dataset_qc.py`, `scripts/train.py`, and `scripts/evaluate.py` exit with `MissingBraTSDataError` if no BraTS-like NIfTI cases are found. The inventory records the layout that was actually present; it does not assume a single archive structure.

Synthetic MRI is not used for reported Dice, HD95, or ECE. Tiny NIfTI files in unit tests are fixtures for loader/validator tests only.

Obtain data from the challenge organizers, accept their license, and do not commit `.nii.gz` files.

## Optional GPU stack

```powershell
.\.venv\Scripts\python -m pip install -e ".[research]"
```

Transformer variants (`unetr`, `swinunetr`) raise `NotImplementedError` until a matched H3 comparison is approved.
