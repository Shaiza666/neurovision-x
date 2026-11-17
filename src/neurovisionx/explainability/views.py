"""Orthogonal slices of 3D arrays.

Axes follow the RAS tensor used in this project after resampling: axis 0 is
treated as axial, 1 coronal, 2 sagittal. These are array axes, not DICOM
instance numbers or scanner-frame labels.
"""

from __future__ import annotations

from typing import Literal

import numpy as np

Orientation = Literal["axial", "coronal", "sagittal"]

ORIENTATION_AXIS: dict[Orientation, int] = {
    "axial": 0,
    "coronal": 1,
    "sagittal": 2,
}

ORIENTATION_ORDER: tuple[Orientation, ...] = ("axial", "coronal", "sagittal")


def slice_index(volume: np.ndarray, orientation: Orientation, index: int | None = None) -> int:
    axis = ORIENTATION_AXIS[orientation]
    length = int(np.asarray(volume).shape[axis])
    if index is None:
        return length // 2
    if index < 0 or index >= length:
        raise IndexError(f"slice {index} out of range for {orientation} (n={length})")
    return int(index)


def extract_slice(
    volume: np.ndarray,
    orientation: Orientation,
    index: int | None = None,
) -> np.ndarray:
    """2D slice. ``volume`` is (D, H, W) or (C, D, H, W) — channel 0 kept if 4D."""
    array = np.asarray(volume)
    if array.ndim == 4:
        array = array[0]
    if array.ndim != 3:
        raise ValueError("expected a 3D volume (or C,D,H,W)")
    axis = ORIENTATION_AXIS[orientation]
    loc = slice_index(array, orientation, index)
    plane = np.take(array, loc, axis=axis)
    if orientation == "axial":
        return np.asarray(plane)
    if orientation == "coronal":
        return np.asarray(np.rot90(plane, k=1))
    return np.asarray(np.rot90(plane, k=1))


def anatomical_label(orientation: Orientation, index: int, length: int) -> str:
    return f"{orientation}  axis={ORIENTATION_AXIS[orientation]}  slice={index}/{length - 1}"
