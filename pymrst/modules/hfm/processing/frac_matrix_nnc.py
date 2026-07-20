"""Matrix-fracture NNC extraction (MRST ``frac_matrix_nnc.m`` layout)."""

from typing import Tuple

import numpy as np

from pymrst.modules.hfm.edfm2d import EDFMGrid, build_edfm


def frac_matrix_nnc(e: EDFMGrid) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return only matrix-fracture NNC pairs, transmissibilities, and CI."""
    mask = np.asarray(e.conn_type) == "frac_matrix"
    return e.conn[mask].copy(), e.trans[mask].copy(), e.ci[mask].copy()


__all__ = ["EDFMGrid", "build_edfm", "frac_matrix_nnc"]
