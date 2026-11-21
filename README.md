# NeuroVision-X

Reproducible **3D glioma MRI segmentation** research stack (PyTorch / MONAI), plus a preserved local **YOLOv8n** detector.

[Documentation](docs/README.md) · [Dataset](docs/dataset.md) · [Training](docs/training.md) · [Evaluation](docs/evaluation.md) · [Research report](docs/research_report.md) · [Architecture](docs/architecture.md)

**This software is an experimental research prototype. It is not a clinically validated diagnostic system. Do not use it for diagnosis or treatment decisions.**

It is **not** a published paper. No conference acceptance, award, or BraTS leaderboard entry is claimed.

## Status (honest)

| Item | State in this repository |
| --- | --- |
| Official BraTS / Synapse NIfTI | **Absent** (`data/raw/brats/` is a README). Not a leaderboard entry. |
| Open data path | **MSD Task01_BrainTumour** (CC-BY-SA 4.0, BraTS 2016/2017-derived). Script: `scripts/download_msd_task01.py`. 484 labeled cases converted on this VM (MD5 matches the MONAI 1.3 digest). Volumes are gitignored. |
| Measured A / B / C | **Yes, on a reduced CPU subset of MSD**, not on official BraTS. See the table below and [`docs/experimental_results.md`](docs/experimental_results.md). |
| Protocol | 4 vCPU, ~16 GB RAM, **no GPU**, PyTorch 2.15.0+cpu. 2 mm, 96³, widths `[8, 16, 32, 64]`, 25 epochs, lr 1e-3, seed 2026, **48 / 8 / 12** patients (prefix of the 339 / 48 / 97 split). Details: [`docs/msd_protocol.md`](docs/msd_protocol.md). |
| A1, A3, A4 and C-minus ablations | **Not run** |
| Hypotheses H1 / H2 | **Not isolated.** C mixes SE, the ET boundary term, and MC dropout. H2's Spearman on C is not a significant negative association on this split. |
| H3 (CNN vs transformer) | **Deferred** (no CUDA) |
| Weights | [GitHub Release `v0.1-msd-cpu`](https://github.com/Shaiza666/neurovision-x/releases/tag/v0.1-msd-cpu). `python scripts/fetch_checkpoints.py` downloads them into `experiments/<run>/best.pt` and checks SHA-256. Not committed as git blobs. Not clinically validated. |
| Unit tests | `python -m unittest discover -s tests` |
| Research dashboard | `app_research.py` loads these checkpoints. Predictions are the checkpoint forward. |

Details: [`docs/experimental_results.md`](docs/experimental_results.md).

## Abstract

NeuroVision-X implements a **matched protocol** for 3D U-Net and Attention U-Net, optional Squeeze-and-Excitation fusion, a Kervadec boundary term on enhancing tumor, Monte Carlo dropout, nested WT/TC/ET metrics, calibration scores, and Grad-CAM overlays. Every modelling piece is an **established** method. The numbers below are a **reduced CPU experiment on MSD Task01**, not a licensed BraTS 2021+ study and not a clinical validation. Full abstract: [`docs/abstract.md`](docs/abstract.md).

## Architecture

The figure below is the **implemented** research workflow. It is not a claim that BraTS training or evaluation has been run.

![NeuroVision-X implemented research pipeline](assets/architecture.png)

Module map: [`docs/architecture.md`](docs/architecture.md).

## MRI examples

Official BraTS NIfTI volumes are **not** in this repository (`data/raw/brats/` is a README only). The pipeline expects four co-registered channels: T1, T1ce, T2, and FLAIR. **No genuine patient MRI is shown here**, because none is stored in git.

![Expected MRI input channels — empty placeholders, not scans](assets/mri_input_layout.png)

## Model architecture

Baseline backbone: **3D U-Net** (Çiçek et al., 2016) as implemented in `src/neurovisionx/models/unet3d.py`, with `in_channels=4`, `out_channels=4` (background, NCR, ED, ET), and YAML features `[16, 32, 64, 128]`. Attention U-Net and an optional SE stem are config alternatives. This figure documents the **code**, not a trained BraTS model.

![Implemented 3D U-Net from configs/model.yaml](assets/model_architecture.png)

A smoke checkpoint trained on **random tensors** (2 epochs, CPU, features `[8, 16]`) may exist locally. That is a software check, not this architecture trained on MRI.

## Segmentation results

Held-out test means on the **reduced MSD CPU cohort** (12 patients, seed 2026). Cells are `mean ± sample std [95% t interval] (n)` from `format_summary` in `src/neurovisionx/evaluation/summarize.py`. HD95 is in **voxels of the 2 mm grid** (n=11 for ET because `test/BRATS_026` has no enhancing-tumor surface). A negative interval bound is the t interval, not a negative distance. A is deterministic T=1. B is deterministic T=1. C is MC dropout **T=10** mean softmax. This is not official BraTS and not a claim that C is better.

| Comparison | Dice WT | Dice TC | Dice ET | HD95 WT (voxels) | HD95 ET (voxels) | ECE |
| --- | --- | --- | --- | --- | --- | --- |
| A 3D U-Net | 0.528 ± 0.238 [0.377, 0.679] (n=12) | 0.583 ± 0.202 [0.454, 0.711] (n=12) | 0.464 ± 0.281 [0.285, 0.642] (n=12) | 26.575 ± 11.197 [19.461, 33.689] (n=12) | 31.378 ± 10.962 [24.014, 38.742] (n=11) | 0.109 ± 0.014 [0.100, 0.118] (n=12) |
| B Attention U-Net | 0.662 ± 0.190 [0.541, 0.783] (n=12) | 0.729 ± 0.149 [0.634, 0.824] (n=12) | 0.680 ± 0.257 [0.517, 0.843] (n=12) | 9.203 ± 7.775 [4.263, 14.143] (n=12) | 7.933 ± 11.982 [-0.117, 15.983] (n=11) | 0.075 ± 0.005 [0.073, 0.078] (n=12) |
| C SE + ET boundary + MC | 0.690 ± 0.154 [0.592, 0.788] (n=12) | 0.740 ± 0.143 [0.650, 0.831] (n=12) | 0.638 ± 0.253 [0.477, 0.799] (n=12) | 13.176 ± 10.727 [6.360, 19.991] (n=12) | 4.441 ± 6.099 [0.344, 8.539] (n=11) | 0.081 ± 0.004 [0.079, 0.083] (n=12) |

On this single split, B has the highest ET Dice and C has the lowest ET HD95 in voxels. C's Spearman ρ (MC entropy vs ET Dice) is **+0.210** (p=0.513, n=12), so H2 is not supported here. Small-ET quartile is only 3 patients. Full table, Brier, precision/recall, and curves: [`docs/experimental_results.md`](docs/experimental_results.md).

![Mean test Dice on the reduced MSD cohort](docs/figures/msd_cpu/dice_means.png)

![Attention U-Net on MSD test/BRATS_422 — FLAIR, prediction, ground truth](docs/figures/msd_cpu/A2_attention_overlay.png)

The overlay is one test case (`test/BRATS_422`), not the mean in the table. Derived millimetres are the unrounded voxel-mean HD95 times 2: A 31.377921594484693 × 2 = 62.755843188969386 mm, B 7.933093602279043 × 2 = 15.866187204558086 mm, C 4.441340618889577 × 2 = 8.882681237779154 mm. Those products are not a second algorithm.

## Weights

The three `best.pt` files (~17 MB each) are release assets, not git blobs:

[MSD CPU checkpoints (`v0.1-msd-cpu`)](https://github.com/Shaiza666/neurovision-x/releases/tag/v0.1-msd-cpu)

```bash
python scripts/fetch_checkpoints.py
```

That writes `experiments/A0_baseline/best.pt`, `experiments/A2_attention/best.pt`, and `experiments/A5_combination/best.pt`, then checks SHA-256 against [`configs/checkpoints.yaml`](configs/checkpoints.yaml). A mismatch deletes the file. These weights are the reduced CPU run above. They are **not** a clinically validated system and **not** official BraTS models.

SHA-256: A `f9fea768469d3e6b21c8535bbf180a7f2db42fb14f44c9c22d02a24c6e80c172`, B `6883c13894b6591fd8c46e7c95e730851fbe249327db3d7093edca6a2d89d1a3`, C `4be1e938f2d2e816a938f7a54a8f43c68cfdf4c43a3baaef03b9bdbffddd1c11`.

## Repository layout

```
assets/            README diagrams (no patient MRI)
configs/           data, model, training, ablations, evaluation YAML
data/              raw / processed / splits (no NIfTI in git)
src/neurovisionx/  3D segmentation package
neurovision_x/     preserved YOLOv8n detector
app.py             YOLO Streamlit UI
app_research.py    3D research dashboard
docs/              report, slides, protocols, bibliography
experiments/       run folders (gitignored weights; README only)
notebooks/
scripts/           train, evaluate, explain, collect_results, …
tests/
```

## Installation

Python ≥ 3.10 (developed on 3.12.10). Full guide: [`docs/installation.md`](docs/installation.md).

```powershell
git clone https://github.com/Shaiza666/neurovision-x.git
cd neurovision-x
py -3.12 -m venv .venv
.\.venv\Scripts\python -m pip install --upgrade pip
.\.venv\Scripts\python -m pip install -e ".[research]"
.\.venv\Scripts\python -m pip install -e ".[app]"
.\.venv\Scripts\python -m unittest
```

## Usage

**Research dashboard** (predictions come only from a loaded `.pt`; no mock logits):

```powershell
.\.venv\Scripts\python -m streamlit run app_research.py
```

If only the smoke checkpoint exists, use the labelled **interface test** (random tensor + real smoke weights). That is not MRI.

**YOLOv8 detector** (unchanged 2D demo):

```powershell
.\.venv\Scripts\python -m streamlit run app.py
```

**Smoke training** (software check, not BraTS):

```powershell
.\.venv\Scripts\python scripts\smoke_train.py
```

**MSD Task01 path** (open download, CC-BY-SA 4.0, not official BraTS). Full steps: [`docs/msd_protocol.md`](docs/msd_protocol.md).

```bash
python scripts/download_msd_task01.py
python scripts/prepare_msd_cpu_cohort.py
python scripts/train.py --ablation A0_baseline --run-name A0_baseline \
    --data-config configs/data_msd_cpu.yaml \
    --model-config configs/model_msd_cpu.yaml \
    --training-config configs/training_msd_cpu.yaml
```

**BraTS path** — exits until licensed volumes sit under `data/raw/brats/` **and** [`docs/experimental_protocol.md`](docs/experimental_protocol.md) is approved:

```powershell
.\.venv\Scripts\python scripts\validate_dataset.py
.\.venv\Scripts\python scripts\prepare_splits.py
.\.venv\Scripts\python scripts\ablation_plan.py
.\.venv\Scripts\python scripts\train.py --ablation A0_baseline --run-name A0_baseline
.\.venv\Scripts\python scripts\evaluate.py --run-name A0_baseline
.\.venv\Scripts\python scripts\collect_results.py
```

Reproducibility notes: [`docs/reproducibility.md`](docs/reproducibility.md). Demonstration walkthrough: [`docs/demo_script.md`](docs/demo_script.md).

## Documentation

| Topic | File |
| --- | --- |
| Full report (intro → conclusion) | [docs/research_report.md](docs/research_report.md) |
| Abstract | [docs/abstract.md](docs/abstract.md) |
| Related work | [docs/literature_review.md](docs/literature_review.md) |
| Gap and hypotheses | [docs/research_gap.md](docs/research_gap.md), [docs/research_hypotheses.md](docs/research_hypotheses.md) |
| Mathematics | [docs/proposed_method.md](docs/proposed_method.md), [docs/metrics.md](docs/metrics.md) |
| Protocol | [docs/experimental_protocol.md](docs/experimental_protocol.md) |
| Architecture | [docs/architecture.md](docs/architecture.md) |
| Install / reproduce | [docs/installation.md](docs/installation.md), [docs/reproducibility.md](docs/reproducibility.md) |
| References (IEEE) | [docs/references.md](docs/references.md) |
| Slides / demo script | [docs/presentation.md](docs/presentation.md), [docs/demo_script.md](docs/demo_script.md) |
| Portfolio summary | [docs/portfolio_summary.md](docs/portfolio_summary.md) |
| Index of all docs | [docs/README.md](docs/README.md) |

## License and citation

MIT. See [LICENSE](LICENSE). Cite the **software repository**, not a fictional article:

```bibtex
@software{neurovisionx2026,
  title        = {NeuroVision-X: a reproducible 3D glioma MRI segmentation research stack},
  author       = {Asghar, Shaiza},
  year         = {2026},
  url          = {https://github.com/Shaiza666/neurovision-x},
  note         = {Experimental prototype. Not a clinically validated diagnostic system.}
}
```

Also [CITATION.cff](CITATION.cff).
