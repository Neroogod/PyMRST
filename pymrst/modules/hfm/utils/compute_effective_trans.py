"""Effective NNC transmissibility (MRST ``computeEffectiveTrans.m`` layout)."""

import numpy as np

from pymrst.modules.hfm.edfm2d import EDFMGrid


def compute_effective_trans(
    ci: np.ndarray | float,
    pv1: np.ndarray | float,
    pv2: np.ndarray | float,
    k1: np.ndarray | float,
    k2: np.ndarray | float,
) -> np.ndarray:
    """Compute MRST's pore-volume weighted harmonic NNC transmissibility."""
    ci = np.asarray(ci, dtype=float)
    pv1 = np.asarray(pv1, dtype=float)
    pv2 = np.asarray(pv2, dtype=float)
    k1 = np.asarray(k1, dtype=float)
    k2 = np.asarray(k2, dtype=float)
    return ci * (pv1 + pv2) / (pv1 / k1 + pv2 / k2)


def nnc_effective_trans(e: EDFMGrid) -> np.ndarray:
    """Return effective transmissibilities for all stored NNC connections."""
    pv = e.poro * e.volumes()
    c1 = e.conn[:, 0]
    c2 = e.conn[:, 1]
    return compute_effective_trans(e.ci, pv[c1], pv[c2], e.perm[c1], e.perm[c2])


__all__ = ["EDFMGrid", "compute_effective_trans", "nnc_effective_trans"]
