# Dataset acquisition and preparation

**NeuroVision-X is an experimental research prototype, not a clinically validated diagnostic system.**

Official BraTS NIfTI is **not** in this repository. Scripts must not download it automatically. `.nii` / `.nii.gz` are gitignored.

## Required release

The loaders accept **BraTS glioma** volumes with four co-registered modalities (T1, T1ce/T1c, T2, FLAIR) and a nested segmentation. Filename tokens:

- 2021-style: `_t1`, `_t1ce`, `_t2`, `_flair`, `_seg`
- 2023+ glioma-style: `-t1n`, `-t1c`, `-t2w`, `-t2f`, `-seg`

**Preferred match for this protocol:** BraTS 2023 Adult Glioma (BraTS-GLI) on Synapse. Organizers state it is the same adult glioma cohort as BraTS 2020/2021 with ET labeled `3` instead of `4`. This repo remaps legacy `4` → `3`.

Do **not** use pediatric, meningioma, or metastasis BraTS tasks as a silent substitute.

## Official source (registration required)

| Item | URL |
| --- | --- |
| Challenge home | [https://www.braintumorsegmentation.org/](https://www.braintumorsegmentation.org/) |
| BraTS 2023 Adult Glioma | [https://www.synapse.org/Synapse:syn51156910](https://www.synapse.org/Synapse:syn51156910) |
| Data folder (request access) | [https://www.synapse.org/Synapse:syn64952532](https://www.synapse.org/Synapse:syn64952532) |
| BraTS-GLI files | [https://www.synapse.org/Synapse:syn51514105](https://www.synapse.org/Synapse:syn51514105) |
| Training zip (after access) | [https://www.synapse.org/Synapse:syn51514132](https://www.synapse.org/Synapse:syn51514132) |
| Post-challenge terms | [https://www.synapse.org/Synapse:syn51156910/wiki/627000](https://www.synapse.org/Synapse:syn51156910/wiki/627000) |

License on the public challenge pages is typically **CC BY-NC 4.0** (non-commercial research) plus the Synapse post-challenge terms. You must accept those terms yourself. This project will not bypass login, CAPTCHA, or license clicks.

## What you must do

1. Create a [Synapse](https://www.synapse.org/) account with a verified email.
2. Open the BraTS 2023 Data folder and click **Request Access**.
3. Read and accept the post-challenge Terms & Conditions.
4. Download **BraTS-GLI** training (and validation if you use official val folders).
5. Unpack so each patient folder sits under `data/raw/brats/` (gitignored).

Example layout (illustrative names only):

```
data/raw/brats/
  BraTS-GLI-xxxxx/
    *-t1n.nii.gz
    *-t1c.nii.gz
    *-t2w.nii.gz
    *-t2f.nii.gz
    *-seg.nii.gz
```

## Prepare (after files exist)

```powershell
.\.venv\Scripts\python scripts\validate_dataset.py
.\.venv\Scripts\python scripts\dataset_qc.py
.\.venv\Scripts\python scripts\prepare_splits.py
```

`prepare_splits.py` writes a patient-level 70/10/20 split (seed 2026) or uses official `training/` / `validation/` folders when those exist. Slices from one patient never appear in two splits.

## Integrity checks the software already performs

- All four modalities required; missing channels are errors, not zeros.
- Empty or unreadable NIfTI raises `CorruptedNiftiError`.
- RAS reorientation, 1 mm resampling, optional foreground crop (`spatial.py`).
- Shape/affine mismatches are resampled to the T1ce (or T1) reference grid.
- Legacy ET label 4 is remapped to 3.

## Storage

The training archive is several tens of gigabytes. This machine recorded ~88 GB free on `C:`. Keep NIfTI **out of git**. Do not commit predictions of real patients either (`experiments/` is gitignored).
