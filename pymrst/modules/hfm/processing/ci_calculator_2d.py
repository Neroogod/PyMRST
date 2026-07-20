"""Conductivity index calculation (MRST ``CIcalculator2D.m`` layout)."""

import numpy as np

from pymrst.gridprocessing.grid import Grid
from pymrst.modules.hfm.edfm2d import FractureLine, _seg_in_cell, avg_frac_distance


def ci_calculator_2d(g: Grid, cell: int, line: FractureLine) -> float:
    """Return CI = fracture length inside cell divided by average distance."""
    seg = _seg_in_cell(g, cell, line)
    if seg is None:
        return 0.0
    p, q = seg
    length = float(np.linalg.norm(q - p))
    if length < 1e-10:
        return 0.0
    return length / avg_frac_distance(g, cell, line)


__all__ = ["FractureLine", "avg_frac_distance", "ci_calculator_2d"]
