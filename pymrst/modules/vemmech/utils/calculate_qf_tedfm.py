"""MRST ``calculateQFTEDFM.m`` TEDFM 2D VEM node weights."""

import numpy as np

from pymrst.modules.vemmech.utils.calculate_qf import calculate_qf


def _matrix_grid(g):
    return getattr(g, "Matrix", g)


def calculate_qf_tedfm(g) -> tuple[np.ndarray, np.ndarray]:
    """Return 2D TEDFM VEM nodal force weights for ``G.Matrix``."""
    return calculate_qf(_matrix_grid(g))

__all__ = ["calculate_qf_tedfm"]
