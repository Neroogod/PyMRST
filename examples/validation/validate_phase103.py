"""Validate 3D EDFM projection and fracture propagation workflows."""

import numpy as np

from pymrst.gridprocessing import cart_grid, compute_geometry
from pymrst.modules.hfm import (
    FractureLine,
    PlanarFracture,
    build_edfm,
    build_edfm_3d,
    project_planar_fracture_to_grid,
    propagate_fractures_by_aperture,
    solve_edfm_tedfm_mechanics,
    solve_incomp_edfm_3d,
)
from pymrst.modules.vemmech import ElasticBC, create_augmented_grid
from pymrst.utils.units import DARCY


def _test_3d_projection_flow():
    g = create_augmented_grid(
        compute_geometry(cart_grid([4, 4, 3], [4.0, 4.0, 3.0]))
    )
    frac = PlanarFracture(
        center=np.array([2.0, 2.0, 1.5]),
        normal=np.array([1.0, 0.0, 0.0]),
        axis1=np.array([0.0, 1.0, 0.0]),
        half_length1=1.5,
        half_length2=1.0,
    )
    proj = project_planar_fracture_to_grid(g, frac, aperture=0.02)
    assert proj.matrix_cells.size > 0
    assert np.all(proj.areas > 0.0)
    assert np.all(np.abs(proj.local_coordinates[:, 0]) <= frac.half_length1 + 1e-12)
    assert np.all(np.abs(proj.local_coordinates[:, 1]) <= frac.half_length2 + 1e-12)

    edfm = build_edfm_3d(
        g,
        [frac],
        aperture=0.02,
        perm_matrix=np.full(g.cells.num, 1.0 * DARCY),
        perm_frac=1.0e4 * DARCY,
        poro_matrix=np.full(g.cells.num, 0.2),
    )
    assert edfm.frac_cells == proj.matrix_cells.size
    assert np.sum(np.asarray(edfm.conn_type) == "frac_matrix") == edfm.frac_cells
    assert np.sum(np.asarray(edfm.conn_type) == "frac") > 0

    left = np.flatnonzero(np.isclose(g.cells.centroids[:, 0], np.min(g.cells.centroids[:, 0])))
    right = np.flatnonzero(np.isclose(g.cells.centroids[:, 0], np.max(g.cells.centroids[:, 0])))
    cells = np.concatenate([left, right])
    tb = np.full(cells.size, 1.0e-10)
    pb = np.concatenate([
        np.full(left.size, 2.0e7),
        np.full(right.size, 1.0e7),
    ])
    p = solve_incomp_edfm_3d(edfm, 1.0e-3, cells, tb, pb)
    assert p.shape == (edfm.n_total,)
    assert np.all(np.isfinite(p))
    assert np.min(p) >= 1.0e7 - 1e-5
    assert np.max(p) <= 2.0e7 + 1e-5
    print(
        "  ok 3D planar EDFM: "
        f"frac_cells={edfm.frac_cells}, nnc={edfm.conn.shape[0]}"
    )


def _propagation_bc(g):
    coords = g.nodes.coords[:, :2]
    bottom = np.flatnonzero(np.isclose(coords[:, 1], 0.0))
    left = np.flatnonzero(np.isclose(coords[:, 0], 0.0))
    disp_nodes = np.concatenate([bottom, left])
    disp_mask = np.vstack([
        np.tile([False, True], (bottom.size, 1)),
        np.tile([True, False], (left.size, 1)),
    ])
    disp_vals = np.zeros((disp_nodes.size, 2))
    bd = np.any(g.faces.neighbors == -1, axis=1)
    ymax = np.max(g.faces.centroids[bd, 1])
    top = np.flatnonzero(bd & np.isclose(g.faces.centroids[:, 1], ymax))
    force = np.tile([0.0, 1.0e5], (top.size, 1))
    return ElasticBC(disp_nodes, disp_mask, disp_vals, top, force)


def _test_2d_fracture_propagation():
    g = compute_geometry(cart_grid([8, 6], [8.0, 6.0]))
    line = FractureLine(np.array([1.5, 3.0, 6.5, 3.0]))
    aperture = 0.02
    perm_m = np.full(g.cells.num, 1.0 * DARCY)
    poro_m = np.full(g.cells.num, 0.2)
    edfm = build_edfm(
        g,
        [line],
        aperture=aperture,
        perm_matrix=perm_m,
        perm_frac=1.0e4 * DARCY,
        poro_matrix=poro_m,
        cell_size=0.5,
    )
    mech = solve_edfm_tedfm_mechanics(
        edfm,
        [line],
        _propagation_bc(g),
        aperture=aperture,
    )
    assert mech.aperture_change[0] > 0.0
    prop = propagate_fractures_by_aperture(
        g,
        [line],
        mech,
        aperture_threshold=0.25 * np.max(mech.aperture_change),
        extension=0.5,
        aperture=aperture,
        perm_matrix=perm_m,
        perm_frac=1.0e4 * DARCY,
        poro_matrix=poro_m,
        cell_size=0.5,
    )
    assert prop.changed
    assert np.allclose(prop.tip_extensions, [[0.5, 0.5]])
    old_len = line.length
    new_len = prop.lines[0].length
    assert np.isclose(new_len, old_len + 1.0)
    assert prop.edfm.frac_cells > edfm.frac_cells
    print(
        "  ok mechanics-driven propagation: "
        f"{old_len:.2f} -> {new_len:.2f}, "
        f"frac_cells={edfm.frac_cells}->{prop.edfm.frac_cells}"
    )


def main():
    print("Phase 103: 3D/projection/fracture-propagation HFM workflows")
    _test_3d_projection_flow()
    _test_2d_fracture_propagation()
    print("Phase 103 validation passed: expanded HFM workflows are stable.")


if __name__ == "__main__":
    main()
