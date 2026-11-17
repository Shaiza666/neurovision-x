# Raw data

Place official BraTS patient folders here after you accept the challenge license.

Do not commit NIfTI files. Do not add synthetic MRI for reported research metrics.

The loader **inspects the tree that is actually present**. It does not assume a
single archive layout. Supported observations include:

- one folder per patient
- official `training/` / `validation/` / `test/` subfolders
- flat NIfTI files grouped by filename prefix

Filename tokens currently recognized:

- 2021-style: `_t1`, `_t1ce`, `_t2`, `_flair`, `_seg`
- 2023+ glioma-style: `-t1n`, `-t1c`, `-t2w`, `-t2f`, `-seg`

Example after a glioma archive is unpacked (illustrative, not a claim that
these files are in git):

```
data/raw/brats/
  BraTS-GLI-xxxxx/
    *-t1n.nii.gz
    *-t1c.nii.gz
    *-t2w.nii.gz
    *-t2f.nii.gz
    *-seg.nii.gz
```

Run `python scripts/validate_dataset.py` to record what was found. If this
directory is empty, scripts exit with `MissingBraTSDataError`.
