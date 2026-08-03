"""Explicit single-fracture stencil (MRST ``explicitStencil.m``)."""

from __future__ import annotations

import numpy as np

from pymrst.gridprocessing.tensor_grid import tensor_grid
from pymrst.utils.geomspace import geomspace


def _transverse_faces(physical_dimensions, half_length, tip_ny, ny):
    front = physical_dimensions[1] / 2 - half_length
    back = physical_dimensions[1] / 2 + half_length
    return np.r_[
        np.linspace(0, front, tip_ny)[:-1],
        np.linspace(front, back, ny)[:-1],
        np.linspace(back, physical_dimensions[1], tip_ny),
    ]


def _fracture_cells(dims, i, j_indices, k_indices):
    nx, ny, _ = map(int, dims)
    return np.array([i + nx*j + nx*ny*k for k in k_indices for j in j_indices], dtype=np.int64)


def explicit_stencil(
    physical_dimensions, *, aperture: float = .02, fracture_spacing: float = 100,
    fracture_half_length: float = 100, nx_refine: int = 10, tip_ny: int = 5,
    ny: int = 11, nz: int = 11, grid_type: str = "geomspacing", **_unused,
):
    """Return ``(grid, fracture_cell_ids, well_cell_ids)`` using zero-based IDs."""
    dims = np.asarray(physical_dimensions, dtype=float).reshape(3)
    kind = grid_type.lower()
    if kind == "geomspacing":
        p = geomspace(aperture/2, fracture_spacing/2, nx_refine, aperture)
    elif kind == "logspacing":
        p = np.logspace(np.log10(aperture/2), np.log10(fracture_spacing/2), nx_refine)
    elif kind == "linspacing":
        p = np.linspace(aperture/2, fracture_spacing/2, nx_refine)
    elif kind == "cartesian":
        x = np.linspace(0, fracture_spacing, 2*nx_refine)
        p = None
    else:
        raise ValueError(f"unknown grid_type {grid_type!r}")
    if p is not None:
        x = fracture_spacing/2 + np.r_[-p[::-1], p]
    y = _transverse_faces(dims, fracture_half_length, tip_ny, ny)
    z = np.linspace(0, dims[2], nz)
    grid = tensor_grid(x, y, z)
    ids = _fracture_cells(grid.cartDims, nx_refine - 1,
                          range(tip_ny - 1, tip_ny + ny - 2), range(nz - 1))
    well = np.array([ids[(ids.size - 1)//2]], dtype=np.int64)
    return grid, ids[None, :], well


explicitStencil = explicit_stencil

__all__ = ["explicit_stencil", "explicitStencil"]
