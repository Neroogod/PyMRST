"""Rock ensemble samples (MRST ``RockSamples.m`` subset)."""

from typing import Optional

import numpy as np


def gaussian_samples(
    mean: np.ndarray,
    cov: np.ndarray,
    n: int,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """Draw an ``(n_par, N)`` Gaussian rock-property ensemble."""
    rng = np.random.default_rng() if rng is None else rng
    mean = np.asarray(mean, float).ravel()
    cov = np.asarray(cov, float)
    if cov.ndim == 1:
        cov = np.diag(cov)
    chol = np.linalg.cholesky(cov)
    return mean[:, None] + chol @ rng.standard_normal((mean.size, n))


__all__ = ["gaussian_samples"]
