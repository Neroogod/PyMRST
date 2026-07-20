"""MRST ``computeLorenz.m`` Lorenz coefficient."""

import numpy as np


def lorenz_coefficient(f: np.ndarray, phi: np.ndarray) -> float:
    """Return Lorenz coefficient from flow/storage capacity curves."""
    dv = np.diff(phi)
    return float(2.0 * (np.sum(0.5 * (f[:-1] + f[1:]) * dv) - 0.5))

__all__ = ["lorenz_coefficient"]
