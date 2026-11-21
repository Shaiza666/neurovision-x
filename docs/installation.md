# Installation and usage

**NeuroVision-X is an experimental research prototype, not a clinically validated diagnostic system.**

Tested locally with Python **3.12.10**, PyTorch **2.15.0+cpu**, MONAI **1.6.1**, Streamlit **1.67**, CUDA **unavailable**. The package metadata requires Python ≥ 3.10. GPU is optional.

Repository: [https://github.com/Shaiza666/neurovision-x](https://github.com/Shaiza666/neurovision-x)

## 1. Clone

```powershell
git clone https://github.com/Shaiza666/neurovision-x.git
cd neurovision-x
```

## 2. Virtual environment

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python -m pip install --upgrade pip
```

On Unix:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

## 3. Install

Core package + tests:

```powershell
.\.venv\Scripts\python -m pip install -e .
```

3D research stack (PyTorch, MONAI, NiBabel, SciPy, Matplotlib):

```powershell
.\.venv\Scripts\python -m pip install -e ".[research]"
```

YOLOv8 Streamlit UI:

```powershell
.\.venv\Scripts\python -m pip install -r requirements.txt
# or
.\.venv\Scripts\python -m pip install -e ".[app]"
```

## 4. Tests

```powershell
.\.venv\Scripts\python -m unittest
```

Shape and loader tests use synthetic tensors or tiny NIfTI fixtures. Those tensors are **not** medical data.

## 5. Command index

| Task | Command | Notes |
| --- | --- | --- |
| Smoke training (not BraTS) | `.\.venv\Scripts\python scripts\smoke_train.py` | Two epochs on random cubes |
| Validate BraTS tree | `.\.venv\Scripts\python scripts\validate_dataset.py` | Exits if no official volumes |
| Patient splits | `.\.venv\Scripts\python scripts\prepare_splits.py` | Patient-level; no slice mix |
| Dataset stats / QC | `scripts\dataset_stats.py`, `scripts\dataset_qc.py` | Same missing-data exit |
| Train (BraTS) | `.\.venv\Scripts\python scripts\train.py --run-name A0_baseline` | Exits until licensed NIfTI exist **and** protocol approval |
| Evaluate | `.\.venv\Scripts\python scripts\evaluate.py --run-name A0_baseline` | Writes `metrics_test.json` from that checkpoint |
| Reliability | `.\.venv\Scripts\python scripts\analyze_reliability.py --run-name A0_baseline --samples 10` | Refuses smoke weights |
| Ablation table | `.\.venv\Scripts\python scripts\ablation_plan.py` | Prints A0–A5; does not train |
| Collect reports | `.\.venv\Scripts\python scripts\collect_results.py` | Rewrites results markdown from JSON |
| Resource check | `.\.venv\Scripts\python scripts\check_resources.py` | CPU/GPU/RAM; does not train |
| H1 JSON compare | `.\.venv\Scripts\python scripts\compare_runs.py --baseline <a> --proposed <b>` | Needs two real `metrics_test.json` files |
| Explain (synthetic layout) | `.\.venv\Scripts\python scripts\explain.py --synthetic` | Watermarked random tensors |
| Explain (patient) | `.\.venv\Scripts\python scripts\explain.py --run-name A0_baseline --patient-id <id>` | Needs non-smoke checkpoint + NIfTI |
| Research dashboard | `.\.venv\Scripts\python -m streamlit run app_research.py` | Real checkpoint only |
| YOLO UI | `.\.venv\Scripts\python -m streamlit run app.py` | Separate detector |

## 6. Data (not shipped)

Official BraTS / Synapse glioma MRI is **not** in git. After accepting the challenge license, unpack volumes under `data/raw/brats/`. See `data/raw/README.md`. Do not commit `.nii.gz`.

Until that tree exists, `train.py` and the dataset scripts raise `MissingBraTSDataError`. That is expected. Step-by-step Synapse access: [dataset.md](dataset.md). Training and evaluation commands: [training.md](training.md), [evaluation.md](evaluation.md).

## 8. Troubleshooting

| Symptom | Cause | Action |
| --- | --- | --- |
| `No BraTS volumes` | `data/raw/brats/` empty | License the data; do not synthesize MRI |
| Smoke checkpoint refused | Evaluating `experiments/smoke/` as research | Train a non-smoke run on official NIfTI |
| CUDA unavailable | CPU PyTorch wheel | Official 96³ jobs need a GPU machine |
| `unetr` NotImplementedError | H3 deferred | Keep 3D U-Net until a matched GPU protocol is approved |
| Streamlit empty views | No inference yet | Load a checkpoint and run a real forward pass |
| Out of memory | 96³ on ~8 GB RAM | Use smoke 16³ locally; run official jobs elsewhere |

## 7. Checkpoints

Smoke weights, if trained locally, live under `experiments/smoke/` and are gitignored. They are software tests, not research models.

The reduced MSD CPU checkpoints (A0, A2, A5) are published at [release `v0.1-msd-cpu`](https://github.com/Shaiza666/neurovision-x/releases/tag/v0.1-msd-cpu). `python scripts/fetch_checkpoints.py` downloads them into `experiments/<run>/best.pt` and checks SHA-256. They are not a clinically validated system and not official BraTS weights.
