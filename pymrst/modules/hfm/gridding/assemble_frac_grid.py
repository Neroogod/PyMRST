"""Fracture-grid assembly (MRST ``assembleFracGrid.m`` layout)."""

from typing import Optional, Sequence

import numpy as np

from pymrst.gridprocessing.grid import Grid
from pymrst.modules.hfm.edfm2d import EDFMGrid, FractureLine, build_edfm


def assemble_frac_grid(
    g: Grid,
    lines: Sequence[FractureLine],
    aperture: float,
    perm_matrix: np.ndarray,
    perm_frac: float,
    poro_matrix: np.ndarray,
    poro_frac: float = 1.0,
    cell_size: float = 0.1,
    n_cells: Optional[Sequence[int]] = None,
    trans_weight: str = "pore_volume",
) -> EDFMGrid:
    """Assemble fracture cells, volumes, centroids, and connections."""
    return build_edfm(
        g,
        lines,
        aperture=aperture,
        perm_matrix=perm_matrix,
        perm_frac=perm_frac,
        poro_matrix=poro_matrix,
        poro_frac=poro_frac,
        cell_size=cell_size,
        n_cells=n_cells,
        trans_weight=trans_weight,
    )


__all__ = ["EDFMGrid", "FractureLine", "assemble_frac_grid", "build_edfm"]
