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


def pressure_bc_contrib_adi_ad(
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

__all__ = [
    "pressure_bc_contrib",
    "pressureBCContrib",
    "pressure_bc_contrib_adi_ad",
]


if __name__ == "__main__":
    from types import SimpleNamespace

    from pymrst.gridprocessing import cart_grid, compute_geometry
    from pymrst.modules.ad_core.utils.bc_to_ad_bc import bc_to_ad_bc
    from pymrst.modules.ad_core.utils.pressure_bc_contrib_adi import pressure_bc_contrib_adi
    from pymrst.params.gravity.gravity import isolated_gravity
    from pymrst.params.wells_and_bc import add_bc

    def test_the_earlier_adi_variant_agrees_with_mrsts_without_gravity():
        G = compute_geometry(cart_grid([2, 1, 1], [2.0, 1.0, 1.0]))
        west = int(np.flatnonzero(np.isclose(G.faces.centroids[:, 0], 0.0))[0])
        bc = bc_to_ad_bc(G, add_bc(None, np.array([west]), "pressure", np.array([12.0])))
        ops = SimpleNamespace(T_all=np.full(G.faces.num, 2.0))
        args = ([np.array([10.0, 8.0])], [np.ones(2)], [np.full(2, 3.0)], [np.ones(2)], bc)
        old, _ = pressure_bc_contrib_adi_ad(G, ops, *args)
        with isolated_gravity("off"):
            new, _ = pressure_bc_contrib_adi(G, ops, *args)
        assert np.allclose(old[0], new[0])
        print("ok test_the_earlier_adi_variant_agrees_with_mrsts_without_gravity")

    test_the_earlier_adi_variant_agrees_with_mrsts_without_gravity()
    print("\nAll tests passed!")
