# Experiment summary (Phase 5)

**Research prototype. Not a clinically validated diagnostic tool.**

Generated: `2026-11-21T14:13:40.293342+00:00`

## What was actually measured

- 3 research-eligible test evaluation(s) from whatever volumes were on disk.
- 0 synthetic smoke run(s), excluded from A/B/C tables.
- A (3D U-Net), B (Attention U-Net), and C (proposed SE + ET boundary + MC dropout) are **not compared on official data** (Synapse BraTS).
- See `docs/experimental_results.md` for the recorded rows. They are not a BraTS leaderboard entry.
- Component ablations C−SE, C−boundary, C−MC stay **not measured** unless a matching non-smoke folder exists.
- Multi-seed means: **not measured** (configured seeds: 2026).

## What is planned, not run

Patient-level evaluation of A/B/C and the three component ablations, with Dice, HD95, precision/recall, small-ET stratum, ECE, Brier, uncertainty–error Spearman/AUPRC, latency, and GPU memory, after protocol approval and licensed BraTS volumes.

## Claims that are not made

The proposed method is **not** reported as more accurate, better calibrated, or faster. Those statements require completed non-smoke `metrics_test.json` files.
