"""Ensemble mean (MRST ``computeMean.m`` layout)."""

import numpy as np


def compute_mean(values, axis=1):
    """Mean across ensemble members by default."""
    return np.asarray(values, float).mean(axis=axis)


__all__ = ["compute_mean"]
