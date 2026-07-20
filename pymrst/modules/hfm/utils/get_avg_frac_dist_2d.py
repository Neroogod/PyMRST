"""Average fracture distance (MRST ``getAvgFracDist2D.m`` layout)."""

from typing import Tuple

import numpy as np

from pymrst.gridprocessing.grid import Grid
from pymrst.modules.hfm.edfm2d import (
    FractureLine,
    _cell_polygon,
    _clip_halfplane,
    _poly_moment,
)


def avg_frac_distance(g: Grid, cell: int, line: FractureLine) -> float:
    """Area-averaged distance from a matrix cell to a fracture line."""
    poly = _cell_polygon(g, cell)
    abc: Tuple[float, float, float] = line.line_coeffs()
    area_tot, moment = 0.0, 0.0
    for keep in (True, False):
        sub = _clip_halfplane(poly, abc, keep)
        area_i, moment_i = _poly_moment(sub, abc)
        area_tot += area_i
        moment += moment_i
    assert area_tot > 0, "degenerate matrix cell"
    return float(moment / area_tot)


def get_avg_frac_dist_2d(g: Grid, cell: int, line: FractureLine) -> float:
    """MRST-style alias for :func:`avg_frac_distance`."""
    return avg_frac_distance(g, cell, line)


__all__ = ["FractureLine", "avg_frac_distance", "get_avg_frac_dist_2d"]
