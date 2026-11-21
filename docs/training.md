# Training guide

**Research prototype. Not a clinically validated diagnostic tool.**  
Do not start the 50-epoch 96³ schedule until licensed BraTS data exist **and** `docs/experimental_protocol.md` is approved.

## Hardware gate (this repository)

```powershell
.\.venv\Scripts\python scripts\check_resources.py
```

This host was recorded as **Python 3.12, PyTorch CPU-only, no NVIDIA driver, ~8 GB RAM, 4 CPU cores**. Official A0–A5 jobs need a **CUDA GPU** and preferably ≥16 GB RAM. Smoke training on 16³ random tensors is the only job that is appropriate here.

## Smoke (software check, not research)

```powershell
.\.venv\Scripts\python scripts\smoke_train.py
```

Writes gitignored `experiments/smoke/{best.pt,latest.pt,history.json}`. Status is `smoke_synthetic_not_research`.

## Official jobs (after data + protocol approval)

Patient splits must already exist (`scripts/prepare_splits.py`).

```powershell
.\.venv\Scripts\python scripts\train.py --run-name A0_baseline --ablation A0_baseline
.\.venv\Scripts\python scripts\train.py --run-name A1_se_fusion --ablation A1_se_fusion
.\.venv\Scripts\python scripts\train.py --run-name A2_attention --ablation A2_attention
.\.venv\Scripts\python scripts\train.py --run-name A3_boundary_et --ablation A3_boundary_et
.\.venv\Scripts\python scripts\train.py --run-name A5_combination --ablation A5_combination
```

**A4 is evaluation-only.** Reuse A0 weights:

```powershell
.\.venv\Scripts\python scripts\analyze_reliability.py --run-name A0_baseline --samples 10
```

Resume:

```powershell
.\.venv\Scripts\python scripts\train.py --run-name A0_baseline --ablation A0_baseline --resume
```

Config: `configs/training.yaml` (AdamW, optional `scheduler: plateau`, AMP when CUDA is present, early stopping on `dice_et`, patience 15). `unetr` / `swinunetr` remain `NotImplementedError` until a matched GPU H3 comparison is approved.
