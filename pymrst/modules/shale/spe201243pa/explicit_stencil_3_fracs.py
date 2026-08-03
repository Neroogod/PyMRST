"""Explicit three-fracture stencil (MRST ``explicitStencil3Fracs.m``)."""

from __future__ import annotations

import numpy as np

from pymrst.gridprocessing.tensor_grid import tensor_grid
from pymrst.utils.geomspace import geomspace
from pymrst.utils.units import FT

from pymrst.modules.shale.spe201243pa.explicit_stencil import _fracture_cells, _transverse_faces


def explicit_stencil_3_fracs(
    physical_dimensions, *, aperture: float = .02, fracture_spacing: float = 100,
    fracture_half_length: float = 100, nx_refine: int = 16,
    nx_refine_small: int = 8, cluster_spacing: float = 65*FT,
    tip_ny: int = 5, ny: int = 11, nz: int = 11,
    grid_type: str = "geomspacing", **_unused,
):
    """Return the three-fracture explicit reference grid and zero-based IDs."""
    dims = np.asarray(physical_dimensions, dtype=float).reshape(3)
    kind = grid_type.lower()
    center = fracture_spacing / 2
    fracture_x = np.array([center - cluster_spacing, center, center + cluster_spacing])

    if kind in {"geomspacing", "cartesian"}:
        small = geomspace(aperture/2, cluster_spacing/2, nx_refine_small, aperture)
        middle = center + np.r_[-small[::-1], small]
        step = aperture if kind == "geomspacing" else 2*aperture
        offset = aperture/2 if kind == "geomspacing" else aperture
        between = geomspace(offset, cluster_spacing/2, nx_refine_small, step)
        left_inner = fracture_x[0] + between[:-1]
        right_inner = fracture_x[2] - between[-2::-1]
        outside = geomspace(
            offset, fracture_spacing/2-cluster_spacing, nx_refine, step,
        )
        x = np.r_[
            fracture_x[0] - outside[::-1], left_inner,
            middle, right_inner, fracture_x[2] + outside,
        ]
        frac_i = np.array([
            nx_refine - 1,
            nx_refine + 2*nx_refine_small - 2,
            nx_refine + 4*nx_refine_small - 3,
        ], dtype=np.int64)
    elif kind in {"logspacing", "linspacing"}:
        p = (np.logspace(np.log10(aperture/2), np.log10(fracture_spacing/2), nx_refine)
             if kind == "logspacing" else
             np.linspace(aperture/2, fracture_spacing/2, nx_refine))
        x = center + np.r_[-p[::-1], p]
        fracture_x = np.array([center])
        frac_i = np.array([nx_refine - 1], dtype=np.int64)
    else:
        raise ValueError(f"unknown grid_type {grid_type!r}")

    if np.any(np.diff(x) <= 0):
        raise ValueError("stencil parameters produced non-monotone x coordinates")
    y = _transverse_faces(dims, fracture_half_length, tip_ny, ny)
    z = np.linspace(0, dims[2], nz)
    grid = tensor_grid(x, y, z)
    j = range(tip_ny - 1, tip_ny + ny - 2)
    k = range(nz - 1)
    rows = np.vstack([_fracture_cells(grid.cartDims, int(i), j, k) for i in frac_i])
    center_index = (rows.shape[1] - 1)//2
    # MRST intentionally returns only the middle-fracture well cell.
    well = np.array([rows[len(rows)//2, center_index]], dtype=np.int64)
    return grid, rows, well


explicitStencil3Fracs = explicit_stencil_3_fracs

__all__ = ["explicit_stencil_3_fracs", "explicitStencil3Fracs"]
