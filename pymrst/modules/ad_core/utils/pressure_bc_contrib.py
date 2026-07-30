"""Legacy pressure-boundary phase contributions."""

from __future__ import annotations

import numpy as np

from pymrst.utils.adi import ad_matmul, value


def _get(obj, name):
    return obj[name] if isinstance(obj, dict) else getattr(obj, name)


def _boundary_cells(grid, faces):
    neighbors = np.asarray(_get(_get(grid, "faces"), "neighbors"), dtype=np.int64)
    selected = neighbors[faces]
    native = np.any(neighbors < 0)
    exterior = selected < 0 if native else selected == 0
    if not np.all(np.sum(exterior, axis=1) == 1):
        raise ValueError("pressure boundary condition is on an internal face")
    cells = np.max(selected, axis=1)
    return cells if native else cells - 1


def _check_pressure_types(boundary):
    kinds = np.atleast_1d(_get(boundary, "type")).astype(str)
    if not np.all(np.char.lower(kinds) == "pressure"):
        raise ValueError("only pressure boundary conditions are allowed")


def pressure_bc_contrib(
    grid,
    operators,
    phase_pressure,
    phase_density,
    phase_mobility,
    phase_bfactor,
    boundary,
    gravity=0.0,
):
    """Cell-indexing version of MRST ``pressureBCContrib``."""
    _check_pressure_types(boundary)
    faces = np.asarray(_get(boundary, "face"), dtype=np.int64).ravel()
    cells = _boundary_cells(grid, faces)
    transmissibility = np.asarray(_get(operators, "T_all"))[faces]
    cell_centroids = np.asarray(_get(_get(grid, "cells"), "centroids"))
    face_centroids = np.asarray(_get(_get(grid, "faces"), "centroids"))
    depth = cell_centroids[cells, -1] - face_centroids[faces, -1]
    target = np.asarray(_get(boundary, "value"), dtype=float).ravel()
    output = []
    for pressure, density, mobility, bfactor in zip(
        phase_pressure, phase_density, phase_mobility, phase_bfactor
    ):
        pressure_bc = pressure[cells]
        density_bc = density[cells]
        difference = target - pressure_bc + gravity * density_bc * depth
        coefficient = bfactor[cells] * mobility[cells]
        active = (np.asarray(value(difference)) <= 0.0).astype(float)
        output.append(-(coefficient * active) * transmissibility * difference)
    return output, cells


def pressure_bc_contrib_adi(
    grid,
    operators,
    phase_pressure,
    phase_density,
    phase_mobility,
    phase_bfactor,
    boundary,
    gravity=0.0,
):
    """Mapping-matrix version of MRST ``pressureBCContribADI``."""
    _check_pressure_types(boundary)
    faces = np.asarray(_get(boundary, "face"), dtype=np.int64).ravel()
    cells = _boundary_cells(grid, faces)
    transmissibility = np.asarray(_get(operators, "T_all"))[faces]
    mapping = _get(boundary, "cell2bcface")
    cell_centroids = np.asarray(_get(_get(grid, "cells"), "centroids"))
    face_centroids = np.asarray(_get(_get(grid, "faces"), "centroids"))
    depth = cell_centroids[cells, -1] - face_centroids[faces, -1]
    target = np.asarray(_get(boundary, "value"), dtype=float).ravel()
    output = []
    for pressure, density, mobility, bfactor in zip(
        phase_pressure, phase_density, phase_mobility, phase_bfactor
    ):
        pressure_bc = ad_matmul(mapping, pressure)
        density_bc = ad_matmul(mapping, density)
        difference = target - pressure_bc + gravity * density_bc * depth
        coefficient = ad_matmul(mapping, bfactor) * ad_matmul(mapping, mobility)
        output.append(-coefficient * transmissibility * difference)
    return output, cells


pressureBCContrib = pressure_bc_contrib
pressureBCContribADI = pressure_bc_contrib_adi

__all__ = [
    "pressure_bc_contrib",
    "pressureBCContrib",
    "pressure_bc_contrib_adi",
    "pressureBCContribADI",
]
