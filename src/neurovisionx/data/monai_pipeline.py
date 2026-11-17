"""Optional MONAI training transforms.

NiBabel validation/QC remains the source of metadata. MONAI is used for the
batched training dictionary pipeline when installed (LoadImaged, Orientationd,
Spacingd, CropForegroundd, NormalizeIntensityd, RandCropByPosNegLabeld).
These are established MONAI recipes, not a new architecture.
"""

from __future__ import annotations

from typing import Any, Sequence

from neurovisionx.data.inventory import PatientCase
from neurovisionx.data.io import IMAGE_MODALITIES


def require_monai() -> Any:
    try:
        import monai
    except ImportError as exc:
        raise ImportError(
            'MONAI is required for this pipeline. Install with: pip install -e ".[research]"'
        ) from exc
    return monai


def case_record(case: PatientCase) -> dict[str, Any]:
    record = {"patient_id": case.patient_id}
    record.update(case.path_map())
    return record


def build_monai_transforms(
    *,
    image_keys: Sequence[str] = IMAGE_MODALITIES,
    has_seg: bool = True,
    target_spacing_mm: tuple[float, float, float] = (1.0, 1.0, 1.0),
    patch_size: tuple[int, int, int] = (96, 96, 96),
    crop_foreground: bool = True,
    training: bool = True,
    foreground_prob: float = 0.67,
) -> Any:
    """Return a MONAI Compose. Raises if MONAI is not installed."""
    require_monai()
    from monai.transforms import (
        Compose,
        CropForegroundd,
        EnsureChannelFirstd,
        LoadImaged,
        NormalizeIntensityd,
        Orientationd,
        RandCropByPosNegLabeld,
        Spacingd,
        SpatialPadd,
    )

    keys = list(image_keys) + (["seg"] if has_seg else [])
    pixdim = tuple(float(v) for v in target_spacing_mm)
    transforms: list[Any] = [
        LoadImaged(keys=keys, image_only=False),
        EnsureChannelFirstd(keys=keys),
        Orientationd(keys=keys, axcodes="RAS"),
        Spacingd(
            keys=keys,
            pixdim=pixdim,
            mode=tuple(["bilinear"] * len(image_keys) + (["nearest"] if has_seg else [])),
        ),
    ]
    if crop_foreground:
        transforms.append(CropForegroundd(keys=keys, source_key=list(image_keys)[0]))
    transforms.append(
        NormalizeIntensityd(keys=list(image_keys), nonzero=True, channel_wise=True)
    )
    if has_seg and training:
        transforms.extend(
            [
                SpatialPadd(keys=keys, spatial_size=patch_size),
                RandCropByPosNegLabeld(
                    keys=keys,
                    label_key="seg",
                    spatial_size=patch_size,
                    pos=foreground_prob,
                    neg=max(1.0 - foreground_prob, 1e-3),
                    num_samples=1,
                ),
            ]
        )
    elif training:
        transforms.append(SpatialPadd(keys=keys, spatial_size=patch_size))
    return Compose(transforms)


def build_monai_dataset(records: list[dict[str, Any]], transform: Any) -> Any:
    require_monai()
    from monai.data import Dataset

    return Dataset(data=records, transform=transform)
