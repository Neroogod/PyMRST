"""Validate HFM v2/DFM/shale compatibility and toughness propagation."""

import numpy as np

from pymrst.gridprocessing import cart_grid, compute_geometry
from pymrst.modules.hfm import (
    FractureLine,
    build_edfm,
    fracture_tip_mode_i_indicator,
    propagate_fractures_by_toughness,
    solve_edfm_tedfm_mechanics,
)
from pymrst.modules.vemmech import ElasticBC
from pymrst.utils.units import DARCY


def _bc(g):
    coords = g.nodes.coords[:, :2]
    bottom = np.flatnonzero(np.isclose(coords[:, 1], 0.0))
    left = np.flatnonzero(np.isclose(coords[:, 0], 0.0))
    nodes = np.concatenate([bottom, left])
    mask = np.vstack([
        np.tile([False, True], (bottom.size, 1)),
        np.tile([True, False], (left.size, 1)),
    ])
    vals = np.zeros((nodes.size, 2))
    bd = np.any(g.faces.neighbors == -1, axis=1)
    top = np.flatnonzero(
        bd & np.isclose(g.faces.centroids[:, 1], np.max(g.faces.centroids[bd, 1]))
    )
    return ElasticBC(nodes, mask, vals, top, np.tile([0.0, 1.0e5], (top.size, 1)))


def test_import_surfaces():
    from pymrst.modules import dfm, hfm_v2, shale

    assert hasattr(hfm_v2, "build_edfm_3d")
    assert hasattr(dfm, "build_dfm")
    assert hasattr(shale, "build_shale_edfm")


def test_toughness_propagation():
    g = compute_geometry(cart_grid([8, 6], [8.0, 6.0]))
    line = FractureLine(np.array([1.5, 3.0, 6.5, 3.0]))
    aperture = 0.02
    E = 1.0e9
    nu = 0.25
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
        edfm, [line], _bc(g), young=E, poisson=nu, aperture=aperture
    )
    k = fracture_tip_mode_i_indicator([line], mech, young=E, poisson=nu)
    assert np.all(k > 0.0)
    prop = propagate_fractures_by_toughness(
        g,
        [line],
        mech,
        toughness=0.25 * float(np.min(k)),
        young=E,
        poisson=nu,
        extension=0.5,
        aperture=aperture,
        perm_matrix=perm_m,
        perm_frac=1.0e4 * DARCY,
        poro_matrix=poro_m,
        cell_size=0.5,
    )
    assert prop.changed
    assert np.allclose(prop.tip_extensions, [[0.5, 0.5]])
    assert prop.driving_force.shape == (1, 2)
    assert prop.edfm.frac_cells > edfm.frac_cells


def main():
    print("Phase 106: HFM v2/DFM/shale compatibility and propagation")
    test_import_surfaces()
    print("  ok compatibility import surfaces")
    test_toughness_propagation()
    print("  ok toughness-driven fracture propagation")
    print("Phase 106 validation passed: HFM workflow surfaces are stable.")


if __name__ == "__main__":
    main()
