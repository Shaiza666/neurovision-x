# MSD Task01 path and the reduced CPU protocol

**Research prototype. Not a clinically validated diagnostic system.**

Official BraTS still lives at `data/raw/brats` and `configs/data.yaml`. This document is the additional, openly downloadable path. It is **not** a Synapse BraTS 2021+ download and **not** a leaderboard submission.

## Source

| Item | Value |
| --- | --- |
| Dataset | Medical Segmentation Decathlon Task01_BrainTumour |
| Derivation | BraTS 2016/2017 (Decathlon paper; not the 2021+ challenge set) |
| License | CC-BY-SA 4.0 |
| Mirror used by the script | `https://msd-for-monai.s3-us-west-2.amazonaws.com/Task01_BrainTumour.tar` |
| MONAI 1.3 MD5 | `240a19d752f0d9e9101544901065d872` (checked; a mismatch is recorded, not hidden) |

Cite Simpson et al. (arXiv:1902.09063, 2019) and Antonelli et al. (*Nature Communications*, 2022) for the Decathlon, and the BraTS organizers for the underlying glioma cohort. Do not cite these CPU numbers as a BraTS challenge score.

## Channel and label mapping

Decathlon `dataset.json` is read at convert time. The published Task01 maps are:

| MSD channel | Name in `dataset.json` | File this repo writes |
| --- | --- | --- |
| 0 | FLAIR | `{id}_flair.nii.gz` |
| 1 | T1w | `{id}_t1.nii.gz` |
| 2 | T1gd | `{id}_t1ce.nii.gz` |
| 3 | T2w | `{id}_t2.nii.gz` |

This is **not** the BraTS stack order (T1, T1ce, T2, FLAIR). The converter assigns channels by name.

| MSD label | Decathlon name | Integer stored for this repo |
| --- | --- | --- |
| 0 | background | 0 |
| 1 | edema | 2 (ED) |
| 2 | non-enhancing tumor | 1 (NCR) |
| 3 | enhancing tumour | 3 (ET) |

Nested scores then follow the existing code: WT = {1,2,3}, TC = {1,3}, ET = {3}. Legacy BraTS label 4 is **not** produced. A `dataset.json` whose names do not match this table raises `MSDSchemaError` instead of being remapped.

Filenames use `_t1` / `_t1ce` / `_t2` / `_flair` / `_seg` so the existing loader works. That token set is a file convention. The voxels are still MSD.

## Commands

```bash
python scripts/download_msd_task01.py
python scripts/validate_dataset.py --config configs/data_msd.yaml
python scripts/prepare_msd_cpu_cohort.py
python scripts/train.py --ablation A0_baseline --run-name A0_baseline \
    --data-config configs/data_msd_cpu.yaml \
    --model-config configs/model_msd_cpu.yaml \
    --training-config configs/training_msd_cpu.yaml
python scripts/train.py --ablation A2_attention --run-name A2_attention \
    --data-config configs/data_msd_cpu.yaml \
    --model-config configs/model_msd_cpu.yaml \
    --training-config configs/training_msd_cpu.yaml
python scripts/train.py --ablation A5_combination --run-name A5_combination \
    --data-config configs/data_msd_cpu.yaml \
    --model-config configs/model_msd_cpu.yaml \
    --training-config configs/training_msd_cpu.yaml
python scripts/evaluate.py --run-name A0_baseline
python scripts/evaluate.py --run-name A2_attention
python scripts/evaluate.py --run-name A5_combination
python scripts/collect_results.py
```

`download_msd_task01.py` writes `data/raw/msd_task01/{patient}/` (gitignored) and `data/splits/msd_task01_patients.json` (patient ids only; seed 2026; 70/10/20).

`prepare_msd_cpu_cohort.py` writes the reduced cohort under `data/raw/msd_task01_cpu/{training,validation,test}/` and `docs/msd_cpu_protocol.json` (ids, shapes, label retention). The official subfolders keep those patients from being shuffled into a new split.

## What the CPU experiment changes

Recorded machine: 4 vCPU, about 16 GB RAM, no CUDA. PyTorch CPU.

| Knob | Full config (`configs/training.yaml`, `configs/model.yaml`) | CPU MSD run |
| --- | --- | --- |
| Data | Licensed BraTS, 1 mm | MSD Task01 subset, resampled to **2 mm** |
| Grid | Native ~240×240×155, patches 96³ | Foreground crop, then crop/pad to **96³** (one volume = one patch) |
| Affine | Scanner / RAS resample | Diagonal 2 mm. **Not** the original scanner pose |
| Widths | `[16, 32, 64, 128]` | `[8, 16, 32, 64]` (same depth, half width) |
| Backbone code | in-repo or MONAI | in-repo only (`optional_monai: false`) |
| Epochs | 50 | **25**, early stop patience 10 on patch `dice_et` |
| AdamW lr | 1e-4 | **1e-3** |
| Cases | all labeled patients | **prefix** of the seed-2026 split: 48 train / 8 val / 12 test |
| AMP | on when CUDA exists | off |
| A5 test inference | MC dropout T=10 | MC dropout T=10 (mean softmax). A and B stay T=1 |

HD95 from `scripts/evaluate.py` is in **voxels** of the 2 mm grid. Multiply by 2 only if you label the product as millimetres; the JSON field stays in voxels.

A0 is the 3D U-Net. A2 is the Attention U-Net. A5 adds the SE stem, the Kervadec boundary term on ET (weight 0.1), and MC dropout at test. A1/A3/A4 and the C-minus ablations are not part of this CPU run unless a later folder exists. H1 and H2 are therefore **not** isolated. H3 (transformer) stays deferred.

## Weights

The trained `best.pt` files are GitHub Release assets, tag `v0.1-msd-cpu`:

https://github.com/Shaiza666/neurovision-x/releases/tag/v0.1-msd-cpu

```bash
python scripts/fetch_checkpoints.py
```

The script downloads each URL in `configs/checkpoints.yaml` to `experiments/<run>/best.pt` and deletes the file if the SHA-256 does not match. They are not committed as git blobs. They are not a clinically validated system. Training commands above reproduce the same protocol if you do not want the published weights.
