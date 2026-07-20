"""Fracture-fracture NNC extraction (MRST ``frac_frac_nnc.m`` layout)."""

from typing import Tuple

import numpy as np

from pymrst.modules.hfm.edfm2d import EDFMGrid, build_edfm


def frac_frac_nnc(e: EDFMGrid) -> Tuple[np.ndarray, np.ndarray]:
    """Return only fracture-fracture NNC pairs and transmissibilities."""
    mask = np.asarray(e.conn_type) == "frac_frac"
    return e.conn[mask].copy(), e.trans[mask].copy()


__all__ = ["EDFMGrid", "build_edfm", "frac_frac_nnc"]
