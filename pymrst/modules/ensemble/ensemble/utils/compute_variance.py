"""Ensemble variance (MRST ``computeVariance.m`` layout)."""

import numpy as np


def compute_variance(values, axis=1, ddof=1):
    """Sample variance across ensemble members by default."""
    return np.asarray(values, float).var(axis=axis, ddof=ddof)


__all__ = ["compute_variance"]
