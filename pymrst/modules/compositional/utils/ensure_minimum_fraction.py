"""Composition fraction floor (MRST ``ensureMinimumFraction.m``)."""

from __future__ import annotations

import numpy as np


def ensure_minimum_fraction(z, tol: float = 1.0e-8):
    """Clamp component fractions to ``tol`` and renormalize each row."""
    arr = np.asarray(z, float)
    floored = np.maximum(arr, tol)
    if floored.ndim == 1:
        total = floored.sum()
    else:
        total = floored.sum(axis=1, keepdims=True)
    return floored / total, tol


def ensureMinimumFraction(z, tol: float = 1.0e-8):
    """MRST camelCase alias for :func:`ensure_minimum_fraction`."""
    return ensure_minimum_fraction(z, tol)


__all__ = ["ensureMinimumFraction", "ensure_minimum_fraction"]
