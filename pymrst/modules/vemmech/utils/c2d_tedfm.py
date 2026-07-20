"""MRST ``C2DTEDFM.m`` TEDFM tensor scaling helper."""

import numpy as np

from pymrst.modules.vemmech.utils.c2d import c2d


def c2d_tedfm(C: np.ndarray, g=None) -> np.ndarray:
    """Scale TEDFM mechanical elasticity tensors.

    MRST's TEDFM mechanics routines read topology from ``G.Matrix``, but the
    tensor scaling itself is identical to ``C2D``. The grid argument is kept
    for API parity.
    """
    return c2d(C)

__all__ = ["c2d_tedfm"]
