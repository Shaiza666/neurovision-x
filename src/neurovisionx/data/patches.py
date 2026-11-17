"""3D patch extraction for volumetric MRI.

Random and grid patches follow standard BraTS / nnU-Net practice. Foreground
bias samples extra patches around tumor voxels so ET is not drowned by
background. This is configuration, not a new architecture.
"""

from __future__ import annotations

import numpy as np

from neurovisionx.data.preprocess import pad_to_min_shape


def _pad_to_patch(
    image: np.ndarray,
    seg: np.ndarray,
    patch_size: tuple[int, int, int],
) -> tuple[np.ndarray, np.ndarray]:
    return pad_to_min_shape(image, seg, patch_size)


def grid_starts(length: int, patch: int, step: int) -> list[int]:
    if length <= patch:
        return [0]
    starts = list(range(0, length - patch + 1, step))
    last = length - patch
    if starts[-1] != last:
        starts.append(last)
    return starts


def extract_grid_patches(
    image: np.ndarray,
    seg: np.ndarray,
    patch_size: tuple[int, int, int],
    overlap: float = 0.0,
) -> list[tuple[np.ndarray, np.ndarray, tuple[int, int, int]]]:
    """Sliding-window patches. ``image`` is (C, D, H, W)."""
    image, seg = _pad_to_patch(image, seg, patch_size)
    pz, py, px = (int(v) for v in patch_size)
    step = tuple(max(int(size * (1.0 - overlap)), 1) for size in (pz, py, px))
    patches = []
    for z in grid_starts(image.shape[1], pz, step[0]):
        for y in grid_starts(image.shape[2], py, step[1]):
            for x in grid_starts(image.shape[3], px, step[2]):
                sl = (slice(None), slice(z, z + pz), slice(y, y + py), slice(x, x + px))
                patches.append((image[sl], seg[z : z + pz, y : y + py, x : x + px], (z, y, x)))
    return patches


def extract_training_patch(
    image: np.ndarray,
    seg: np.ndarray,
    patch_size: tuple[int, int, int],
    rng: np.random.Generator,
    foreground_prob: float = 0.67,
) -> tuple[np.ndarray, np.ndarray]:
    """One patch, optionally centered on a non-background label voxel."""
    image, seg = _pad_to_patch(image, seg, patch_size)
    # If we padded, volume may already equal patch size.
    if image.shape[1:] == tuple(int(v) for v in patch_size):
        return image, seg

    pz, py, px = (int(v) for v in patch_size)
    depth, height, width = image.shape[1:]
    use_fg = float(rng.random()) < foreground_prob and np.any(seg > 0)
    if use_fg:
        coords = np.argwhere(seg > 0)
        z, y, x = coords[int(rng.integers(0, len(coords)))]
        z0 = int(np.clip(z - pz // 2, 0, depth - pz))
        y0 = int(np.clip(y - py // 2, 0, height - py))
        x0 = int(np.clip(x - px // 2, 0, width - px))
    else:
        z0 = int(rng.integers(0, depth - pz + 1))
        y0 = int(rng.integers(0, height - py + 1))
        x0 = int(rng.integers(0, width - px + 1))
    sl = (slice(None), slice(z0, z0 + pz), slice(y0, y0 + py), slice(x0, x0 + px))
    return image[sl], seg[z0 : z0 + pz, y0 : y0 + py, x0 : x0 + px]
