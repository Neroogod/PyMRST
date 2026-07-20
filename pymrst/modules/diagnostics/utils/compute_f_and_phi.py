"""MRST ``computeFandPhi.m`` flow-capacity/storage-capacity curve."""

from typing import Tuple

import numpy as np


def compute_f_phi(
    pv: np.ndarray,
    tof_total: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Flow capacity ``F`` against storage capacity ``Phi``."""
    order = np.argsort(tof_total)
    t = tof_total[order]
    v = pv[order]
    phi = np.cumsum(v)
    phi = np.concatenate([[0.0], phi / phi[-1]])
    flux = v / np.maximum(t, 1e-300)
    ff = np.cumsum(flux)
    f = np.concatenate([[0.0], ff / ff[-1]])
    return f, phi

__all__ = ["compute_f_phi"]
