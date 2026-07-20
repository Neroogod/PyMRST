"""MRST ``computeSweep.m`` sweep efficiency curve."""

from typing import Tuple

import numpy as np


def compute_sweep(f: np.ndarray, phi: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Sweep efficiency ``Ev`` against dimensionless time ``tD``."""
    keep = np.ones(f.size, dtype=bool)
    keep[1:] = f[:-1] <= f[1:] - np.sqrt(np.finfo(float).eps)
    f, phi = f[keep], phi[keep]
    td = np.concatenate([[0.0], np.diff(phi) / np.diff(f)])
    ev = phi + (1.0 - f) * td
    return ev, td

__all__ = ["compute_sweep"]
