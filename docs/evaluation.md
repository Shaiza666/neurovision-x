# Evaluation guide

**Research prototype. Not a clinically validated diagnostic tool.**  
Metrics are written only from a **non-smoke** checkpoint and **official** patient NIfTI.

## After a real training run

```powershell
.\.venv\Scripts\python scripts\evaluate.py --run-name A0_baseline --split test
```

Writes (gitignored under `experiments/`):

- `metrics_test.json` — per-patient Dice, HD95, precision, recall/sensitivity, specificity, ECE, NLL, Brier; mean ± std and 95% t-interval
- `metrics_test.csv` — the same rows for spreadsheets

Optional prediction NIfTI (still gitignored):

```powershell
.\.venv\Scripts\python scripts\evaluate.py --run-name A0_baseline --split test --save-preds
```

## Reliability (H2)

```powershell
.\.venv\Scripts\python scripts\analyze_reliability.py --run-name A0_baseline --samples 10
```

Uses MC dropout or flip TTA according to `training.uncertainty.method`. Refuses smoke weights and missing BraTS.

## H1 paired test (needs two real JSON files)

```powershell
.\.venv\Scripts\python scripts\compare_runs.py --baseline experiments/A0_baseline/metrics_test.json --proposed experiments/A3_boundary_et/metrics_test.json --metric hd95_et
```

Wilcoxon signed-rank and paired Cohen's d are computed only on **shared patient IDs** present in both files. The script sets `"supported": false`; you interpret H1 after reading p-values from real data. Missing files exit without a fake table.

## Collect markdown tables

```powershell
.\.venv\Scripts\python scripts\collect_results.py
```

Smoke folders never fill A/B/C cells. Missing JSON stays **not measured**.
