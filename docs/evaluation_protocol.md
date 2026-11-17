# Evaluation protocol (Phase 5)

**Status: awaiting approval for expensive BraTS jobs.** This document is the scientific evaluation plan. `scripts/collect_results.py` writes `docs/experimental_results.md` from **on-disk JSON only**. It does not train and does not invent metrics.

This software is a research prototype, not a clinically validated diagnostic tool.

## Comparison groups

| Label | Run ID | What it is |
| --- | --- | --- |
| A | `A0_baseline` | 3D U-Net, early fusion, Dice+CE |
| B | `A2_attention` | Attention U-Net, early fusion, Dice+CE |
| C | `A5_combination` | Proposed: SE stem + ET boundary + MC dropout \(T=10\) |

All arms share: patient-level split (`configs/data.yaml`), RAS + 1 mm spacing + z-score, patch size from `training.yaml`, seed **2026** unless extra seeds are approved.

## Component ablations (from C)

| Arm | Removes | How |
| --- | --- | --- |
| `C_minus_se` | SE fusion | Reuse `A3_boundary_et` weights; MC \(T=10\) at test |
| `C_minus_boundary` | ET boundary loss | Reuse `A1_se_fusion` weights; MC \(T=10\) |
| `C_minus_mc` | MC dropout | Reuse `A5_combination` weights; \(T=1\) |

No extra training jobs beyond A0, A1, A2, A3, A5.

## Metrics

Computed on the **test** patients of the frozen split. Per-patient scores, then mean, sample SD (ddof=1), and 95% Student-t CI when \(n \ge 2\). Empty surfaces → HD95 NaN (omitted from the mean, not replaced).

- Overlap: Dice WT/TC/ET  
- Surface: HD95 WT/TC/ET  
- Detection: precision and recall WT/TC/ET  
- Small ET: lowest quartile of ET voxel count (pre-registered)  
- Calibration: ECE (15 bins), NLL, multiclass Brier  
- Uncertainty–error: Spearman \(\rho\), AUPRC for worst ET Dice quartile  
- Cost: wall-clock ms (\(T=1\), \(T=10\)), peak CUDA allocated MB, parameter count  

## Seeds

Default: `{2026}`. Extra seeds `{2027, 2028}` only if compute is approved. Do not create empty seed folders.

## Honesty

- Smoke checkpoints are not evaluated as BraTS.  
- `collect_results.py` prints **not measured** when JSON is missing.  
- Do not copy challenge leaderboards into `experiments/` or this report.  
- Unsuccessful jobs stay in the inventory with their real status.

## After approval

Train A0 → A2 → A3 → A1 → A5 as compute allows, then evaluate and collect. See `docs/experimental_protocol.md` (Phase 4) for the training order.
