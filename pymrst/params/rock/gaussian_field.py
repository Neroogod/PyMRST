"""Smooth random fields, ported from MRST `gaussianField`."""

from typing import Optional, Sequence

import numpy as np
from scipy.signal import fftconvolve


def gaussian_field(
    dims: Sequence[int],
    vals: Sequence[float] = (-1.0, 1.0),
    size: int | Sequence[int] = 3,
    std: float = 0.65,
    rng: Optional[np.random.Generator] = None,
    field: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Return a smoothed random field rescaled to `vals`."""
    dims = tuple(int(d) for d in np.atleast_1d(dims))
    if not 1 <= len(dims) <= 3 or any(d <= 0 for d in dims):
        raise ValueError("dims must contain one to three positive sizes")
    values = np.asarray(vals, dtype=float).ravel()
    if values.size != 2:
        raise ValueError("vals must contain exactly two entries")
    if not np.isfinite(std) or std <= 0:
        raise ValueError("std must be positive")
    raw_size = np.asarray(size, dtype=np.int64).ravel()
    if raw_size.size == 1:
        size_arr = np.repeat(raw_size, len(dims))
    elif len(dims) <= raw_size.size <= 3:
        size_arr = raw_size[:len(dims)]
    else:
        raise ValueError("size must be scalar or cover every field dimension")
    if np.any(size_arr <= 0) or np.any(size_arr % 2 == 0):
        raise ValueError("Gaussian kernel sizes must be positive odd integers")
    if field is None:
        rng = np.random.default_rng() if rng is None else rng
        field = rng.standard_normal(dims)
    field = np.asarray(field, dtype=np.float64)
    if field.shape != dims:
        raise ValueError(f"field must have shape {dims}, got {field.shape}")

    smoothed = smooth_field(field, size_arr, std)
    lo, hi = values
    pmin, pmax = smoothed.min(), smoothed.max()
    if np.isclose(pmax, pmin, rtol=32*np.finfo(float).eps, atol=0.0):
        # MRST's literal rescaling divides by zero for a constant draw.
        return np.full(dims, np.nan)
    return lo + (hi - lo) * (smoothed - pmin) / (pmax - pmin)


def gaussian_kernel(size: Sequence[int], std: float) -> np.ndarray:
    """Return an isotropic Gaussian kernel normalized to sum to one."""
    size_arr = np.atleast_1d(np.asarray(size, dtype=np.int64)).ravel()
    if size_arr.size == 0 or np.any(size_arr <= 0) or np.any(size_arr % 2 == 0):
        raise ValueError("Gaussian kernel sizes must be positive odd integers")
    if not np.isfinite(std) or std <= 0:
        raise ValueError("std must be positive")
    pad = (size_arr - 1) // 2
    axes = [np.arange(-p, p + 1) for p in pad]
    grids = np.meshgrid(*axes, indexing="ij")
    radius_squared = sum(grid.astype(float) ** 2 for grid in grids)
    kernel = np.exp(-radius_squared / (2.0 * std * std))
    return kernel / kernel.sum()


def smooth_field(field: np.ndarray, size: Sequence[int], std: float) -> np.ndarray:
    """Smooth with replicated edges, matching MATLAB `smooth3` padding."""
    kernel = gaussian_kernel(size, std)
    pad = [((n - 1) // 2, (n - 1) // 2) for n in kernel.shape]
    padded = np.pad(field, pad, mode="edge")
    return fftconvolve(padded, kernel, mode="valid")


if __name__ == "__main__":

    def roughness(values: np.ndarray) -> float:
        """Return mean neighbor difference along the first axis."""
        return float(np.abs(np.diff(values, axis=0)).mean())

    def test_gaussian_field_is_smooth_and_scaled() -> None:
        rng = np.random.default_rng(0)
        field = gaussian_field((40, 40), vals=(0.1, 0.3), size=7, std=2.0, rng=rng)
        assert field.shape == (40, 40)
        assert np.isclose(field.min(), 0.1)
        assert np.isclose(field.max(), 0.3)
        print("ok test_gaussian_field_is_smooth_and_scaled")

    def test_larger_kernel_smooths_more() -> None:
        rng = np.random.default_rng(1)
        raw = rng.standard_normal((60, 60))
        rough = gaussian_field((60, 60), size=3, std=0.65, field=raw)
        smooth = gaussian_field((60, 60), size=15, std=4.0, field=raw)
        assert roughness(smooth) < 0.5 * roughness(rough)
        print("ok test_larger_kernel_smooths_more")

    def test_gaussian_field_does_not_fade_at_edges() -> None:
        rng = np.random.default_rng(2)
        field = gaussian_field((60, 60), vals=(0.0, 1.0), size=11, std=3.0, rng=rng)
        edge = np.concatenate([field[0], field[-1], field[:, 0], field[:, -1]])
        interior = field[10:-10, 10:-10].ravel()
        assert abs(edge.mean() - interior.mean()) < 0.15
        print("ok test_gaussian_field_does_not_fade_at_edges")

    test_gaussian_field_is_smooth_and_scaled()
    test_larger_kernel_smooths_more()
    test_gaussian_field_does_not_fade_at_edges()
    print("\nAll tests passed!")
