# NeuroVision-X

Research prototype for **3D glioma segmentation on multimodal MRI**, plus a preserved local **YOLOv8n** detection demo.

**This software is not a clinically validated diagnostic tool.** Do not use it for diagnosis or treatment decisions. No BraTS training has been run in this repository unless an `experiments/` run folder contains a real log and checkpoint.

Phase 1 literature and hypotheses: `docs/`. Reproducibility notes: `docs/reproducibility.md`.

## Layout

```
configs/           experiment YAML (data, model, training)
data/              raw / processed / patient splits (no NIfTI in git)
src/neurovisionx/  3D segmentation package
app/               notes; Streamlit detection UI remains at repo-root app.py
docs/              literature review and research plan
experiments/       run metadata (empty until a real training job)
notebooks/
scripts/           validate_dataset, prepare_splits, dataset_stats, dataset_qc, smoke_train, train, evaluate, infer
tests/
```

The YOLO detector under `neurovision_x/` is unchanged.

## 3D research code

```powershell
.\.venv\Scripts\python -m pip install -e .
.\.venv\Scripts\python -m unittest
.\.venv\Scripts\python scripts\train.py
```

`scripts/train.py` **exits** until official BraTS patient folders exist under `data/raw/brats/`. It will not write fake Dice scores. Optional GPU stack: `pip install -e ".[research]"` (PyTorch, MONAI, NiBabel).

Place BraTS volumes only after accepting the challenge license. The Phase 2 loader **inspects the tree on disk** (patient folders, official train/val subfolders, or flat NIfTI groups). 2021-style (`_t1`, `_t1ce`, `_flair`) and 2023+ glioma (`-t1n`, `-t1c`, `-t2f`) filenames are recognized. Nested WT/TC/ET scores are derived from NCR/ED/ET labels; legacy ET label `4` is mapped to `3`.

```powershell
.\.venv\Scripts\python -m pip install -e ".[research]"
.\.venv\Scripts\python scripts\validate_dataset.py
.\.venv\Scripts\python scripts\prepare_splits.py
.\.venv\Scripts\python scripts\dataset_stats.py
.\.venv\Scripts\python scripts\dataset_qc.py
```

No official BraTS archive is currently in this workspace. Those scripts exit with `MissingBraTSDataError` until you unpack a licensed release under `data/raw/brats/`.

## Phase 3 baseline (PyTorch)

3D U-Net (Çiçek 2016) is the default. Attention U-Net (Oktay 2018) is the comparison (`name: attention_unet3d` in `configs/model.yaml`). Dice+CE uses inverse-frequency CE weights for class imbalance. Mixed precision, gradient clipping, `best.pt` / `latest.pt`, validation Dice for WT/TC/ET, and learning curves are in the training engine. HD95 is documented in `docs/metrics.md` and computed on full volumes at evaluation time.

```powershell
.\.venv\Scripts\python -m pip install -e ".[research]"
.\.venv\Scripts\python scripts\smoke_train.py
```

`smoke_train.py` runs **two epochs on random tensors**. Those losses and Dice values are **not research results**. After a licensed BraTS tree is in place:

```powershell
.\.venv\Scripts\python scripts\train.py --run-name unet3d
.\.venv\Scripts\python scripts\train.py --run-name unet3d --resume
.\.venv\Scripts\python scripts\evaluate.py --run-name unet3d
```

## Detection UI (existing)

```powershell
.\.venv\Scripts\python -m pip install -r requirements.txt
.\.venv\Scripts\python -m streamlit run app.py
.\.venv\Scripts\python -m neurovision_x image photo.jpg -o outputs\photo.jpg
```

## Tests

```powershell
.\.venv\Scripts\python -m unittest
```

Split, config, Dice, and 3D U-Net **shape** tests use synthetic tensors where needed. Those tensors are not medical data and are not reported as research metrics.
