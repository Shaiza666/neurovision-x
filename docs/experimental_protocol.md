# Experimental protocol (Phase 4)

**Status: awaiting approval.** Expensive BraTS training must **not** start until this protocol is approved and official data are present.

This software is a research prototype, not a clinically validated diagnostic tool. Synthetic smoke numbers are not results.

## Preconditions

1. Phase 1 hypotheses H1/H2 (H3 deferred unless GPU is approved).  
2. Licensed BraTS glioma volumes under `data/raw/brats/` (layout discovered, not assumed).  
3. Patient-level split written by `scripts/prepare_splits.py` and frozen (`data/splits/patients.json`).  
4. Same seed `2026`, same preprocessing YAML, same patch size for all trained arms.

## Ablation arms

| ID | Train? | Architecture | Fusion | \(\lambda_{\mathrm{bd}}\) | MC \(T\) at test | Isolates |
| --- | --- | --- | --- | --- | --- | --- |
| A0 | yes | 3D U-Net | early | 0 | 1 | Baseline (Phase 3) |
| A1 | yes | 3D U-Net | SE stem | 0 | 1 | Fusion only |
| A2 | yes | Attention U-Net | early | 0 | 1 | Attention only |
| A3 | yes | 3D U-Net | early | \(>0\) on ET | 1 | Boundary only (H1) |
| A4 | no | *reuse A0 weights* | early | 0 | 10 | MC dropout only (H2) |
| A5 | yes | 3D U-Net | SE stem | \(>0\) on ET | 10 | Combination (not claimed SOTA) |

Default \(\lambda_{\mathrm{bd}}=0.1\) if A3/A5 are approved (grid-search only on **val**, never test).

A4 does **not** retrain. Dropout is already in the U-Net; MC is test-time.

## Metrics (test patients, then mean ± std)

- Dice WT, TC, ET  
- HD95 WT, TC, ET (NaN if a surface cannot be formed; see `docs/metrics.md`)  
- ECE (15 bins) and NLL on voxels, documented class handling  
- Spearman \(\rho\) (patient uncertainty vs Dice); \(\alpha=0.05\), two-sided  
- AUPRC for detecting the worst **quartile** of ET Dice  
- ET-size stratum: lowest quartile of ET voxel count vs the rest  
- Parameters; mean inference milliseconds per volume for \(T=1\) and \(T=10\)

## Leakage and honesty

- No slice mixing across patients.  
- No synthetic MRI in reported tables.  
- Unsuccessful runs are logged.  
- Do not copy leaderboard numbers into `experiments/`.

## Compute plan (suggested)

1. Smoke (done in Phase 3): tensor shapes only.  
2. After approval: A0 on real data (longest wall-clock).  
3. A3 (H1) if A0 finishes.  
4. A4 analysis on A0 checkpoint (cheap).  
5. A1/A2/A5 only if compute remains.

## How to start (only after written approval)

```powershell
.\.venv\Scripts\python scripts\ablation_plan.py
# After approval and data:
.\.venv\Scripts\python scripts\train.py --run-name A0_baseline
.\.venv\Scripts\python scripts\evaluate.py --run-name A0_baseline
.\.venv\Scripts\python scripts\analyze_reliability.py --run-name A0_baseline
```

`scripts/ablation_plan.py` prints this table and **does not train**.
