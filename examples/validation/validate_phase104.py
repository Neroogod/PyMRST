"""Validate 3D global elastoplastic VEM coupling."""

import numpy as np

from pymrst.gridprocessing import cart_grid, compute_geometry
from pymrst.modules.ad_mechanics import solve_elastoplastic_vem
from pymrst.modules.vemmech import ElasticBC, create_augmented_grid, enu2c
from pymrst.modules.vemmech.vem import vem_lin_elast


def _case(force=1.5e5):
    g = create_augmented_grid(
        compute_geometry(cart_grid([2, 2, 2], [2.0, 2.0, 2.0]))
    )
    coords = g.nodes.coords[:, :3]
    xmin = np.min(coords[:, 0])
    ymin = np.min(coords[:, 1])
    zmin = np.min(coords[:, 2])
    xmax = np.max(coords[:, 0])
    xnodes = np.flatnonzero(np.isclose(coords[:, 0], xmin))
    ynodes = np.flatnonzero(np.isclose(coords[:, 1], ymin))
    znodes = np.flatnonzero(np.isclose(coords[:, 2], zmin))
    disp_nodes = np.concatenate([xnodes, ynodes, znodes])
    disp_mask = np.vstack([
        np.tile([True, False, False], (xnodes.size, 1)),
        np.tile([False, True, False], (ynodes.size, 1)),
        np.tile([False, False, True], (znodes.size, 1)),
    ])
    disp_vals = np.zeros((disp_nodes.size, 3))
    bd = np.any(g.faces.neighbors == -1, axis=1)
    right_faces = np.flatnonzero(
        bd & np.isclose(g.faces.centroids[:, 0], xmax)
    )
    tr = np.tile([force, 0.0, 0.0], (right_faces.size, 1))
    bc = ElasticBC(disp_nodes, disp_mask, disp_vals, right_faces, tr)
    return g, bc


def main():
    print("Phase 104: 3D global elastoplastic VEM coupling")
    E = 1.0e9
    nu = 0.25
    g, bc = _case()
    C = enu2c(np.full(g.cells.num, E), np.full(g.cells.num, nu))
    u_elastic, _ = vem_lin_elast(
        g, C, bc, load=lambda x: np.zeros_like(x)
    )
    high_yield = solve_elastoplastic_vem(
        g, bc, young=E, poisson=nu, yield_stress=1.0e12,
        load_factors=[1.0],
    )
    assert np.allclose(high_yield.last.displacement, u_elastic, rtol=5e-7, atol=1e-10)
    assert high_yield.last.strain.shape == (g.cells.num, 6)
    assert high_yield.last.stress.shape == (g.cells.num, 6)
    assert not np.any(high_yield.last.plastic_state.yielded)
    print("  ok 3D high-yield limit reproduces elastic VEM solve")

    plastic = solve_elastoplastic_vem(
        g,
        bc,
        young=E,
        poisson=nu,
        yield_stress=4.0e4,
        hardening=1.0e7,
        load_factors=[0.25, 0.5, 0.75, 1.0],
        max_iter=18,
    )
    assert np.any(plastic.last.plastic_state.yielded)
    assert np.max(plastic.last.plastic_state.eq_plastic_strain) > 0.0
    assert plastic.last.residual_norm < 5e-3
    print(
        "  ok 3D plastic load ramp: "
        f"yielded={np.count_nonzero(plastic.last.plastic_state.yielded)}, "
        f"max_eqp={np.max(plastic.last.plastic_state.eq_plastic_strain):.3e}"
    )
    print("Phase 104 validation passed: 3D elastoplastic coupling works.")


if __name__ == "__main__":
    main()
