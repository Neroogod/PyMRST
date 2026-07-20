"""Validate EDFM-to-TEDFM mechanics workflow coupling."""

import numpy as np

from pymrst.gridprocessing import cart_grid, compute_geometry
from pymrst.modules.hfm import (
    FractureLine,
    build_edfm,
    fracture_normal_strain,
    solve_edfm_tedfm_mechanics,
)
from pymrst.modules.vemmech import ElasticBC
from pymrst.utils.units import DARCY


def _bc(g):
    coords = g.nodes.coords[:, :2]
    bottom_nodes = np.flatnonzero(np.isclose(coords[:, 1], 0.0))
    left_nodes = np.flatnonzero(np.isclose(coords[:, 0], 0.0))
    disp_nodes = np.concatenate([bottom_nodes, left_nodes])
    disp_mask = np.vstack([
        np.tile([False, True], (bottom_nodes.size, 1)),
        np.tile([True, False], (left_nodes.size, 1)),
    ])
    disp_vals = np.zeros((disp_nodes.size, 2))

    bd = np.any(g.faces.neighbors == -1, axis=1)
    ymax = np.max(g.faces.centroids[bd, 1])
    top_faces = np.flatnonzero(bd & np.isclose(g.faces.centroids[:, 1], ymax))
    force = np.tile([0.0, -1.0e5], (top_faces.size, 1))
    return ElasticBC(
        disp_nodes=disp_nodes,
        disp_mask=disp_mask,
        disp_vals=disp_vals,
        force_faces=top_faces,
        force=force,
    )


def main():
    print("Phase 99: EDFM/TEDFM mechanics workflow coupling")
    g = compute_geometry(cart_grid([8, 6], [8.0, 6.0]))
    line = FractureLine(np.array([1.0, 3.0, 7.0, 3.0]))
    aperture = 0.02
    edfm = build_edfm(
        g,
        [line],
        aperture=aperture,
        perm_matrix=np.full(g.cells.num, 1.0 * DARCY),
        perm_frac=1.0e4 * DARCY,
        poro_matrix=np.full(g.cells.num, 0.2),
        cell_size=0.5,
    )
    pressure = np.linspace(2.0e6, 1.0e6, edfm.n_total)
    result = solve_edfm_tedfm_mechanics(
        edfm,
        [line],
        _bc(g),
        pressure=pressure,
        aperture=aperture,
    )

    assert result.displacement.shape == (g.nodes.num, 2)
    assert result.stress.shape == (g.cells.num, 3)
    assert result.strain.shape == (g.cells.num, 3)
    assert result.normal_strain.shape == (edfm.frac_cells,)
    assert result.aperture_change.shape == (edfm.frac_cells,)
    assert np.all(np.isfinite(result.displacement))
    assert np.all(np.isfinite(result.normal_strain))
    assert np.allclose(
        result.aperture_change, aperture * result.normal_strain
    )

    normal_strain, rep = fracture_normal_strain(edfm, [line], result.strain)
    assert np.allclose(normal_strain, result.normal_strain)
    assert np.array_equal(rep, result.matrix_cell_for_fracture)
    assert np.all(result.matrix_cell_for_fracture >= 0)
    assert result.extra["disc"]["div"].shape[0] == g.cells.num
    print(
        "  ok coupled solve: "
        f"fracture_cells={edfm.frac_cells}, "
        f"max|da|={np.max(np.abs(result.aperture_change)):.3e}"
    )
    print("Phase 99 validation passed: EDFM feeds TEDFM mechanics fields.")


if __name__ == "__main__":
    main()
