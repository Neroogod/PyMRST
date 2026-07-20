"""MRST ``calculateQCTEDFM.m`` TEDFM 3D VEM node weights."""

from typing import Tuple

import numpy as np

from pymrst.modules.vemmech.utils.calculate_qc import calculate_qc


def _matrix_grid(g):
    return getattr(g, "Matrix", g)


def calculate_qc_tedfm(g) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return 3D TEDFM VEM weights for the mechanical ``G.Matrix`` grid."""
    return calculate_qc(_matrix_grid(g))

__all__ = ["calculate_qc_tedfm"]
