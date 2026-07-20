"""Validate global elastoplastic VEM Newton coupling."""

import numpy as np

from pymrst.gridprocessing import cart_grid, compute_geometry
from pymrst.modules.ad_mechanics import solve_elastoplastic_vem
from pymrst.modules.vemmech import ElasticBC, create_augmented_grid, enu2c_2d
from pymrst.modules.vemmech.vem import vem_lin_elast


def _case(force=2.0e5):
    g = create_augmented_grid(
        compute_geometry(cart_grid([3, 2], [3.0, 2.0]))
    )
    coords = g.nodes.coords[:, :2]
    left = np.flatnonzero(np.isclose(coords[:, 0], 0.0))
    bottom = np.flatnonzero(np.isclose(coords[:, 1], 0.0))
    disp_nodes = np.concatenate([left, bottom])
    disp_mask = np.vstack([
        np.tile([True, False], (left.size, 1)),
        np.tile([False, True], (bottom.size, 1)),
    ])
    disp_vals = np.zeros((disp_nodes.size, 2))
    bd = np.any(g.faces.neighbors == -1, axis=1)
    xmax = np.max(g.faces.centroids[bd, 0])
    right_faces = np.flatnonzero(bd & np.isclose(g.faces.centroids[:, 0], xmax))
    tr = np.tile([force, 0.0], (right_faces.size, 1))
    bc = ElasticBC(disp_nodes, disp_mask, disp_vals, right_faces, tr)
    return g, bc


def main():
    print("Phase 102: global elastoplastic VEM Newton coupling")
    E = 1.0e9
    nu = 0.25
    g, bc = _case()
    C = enu2c_2d(np.full(g.cells.num, E), np.full(g.cells.num, nu))
    u_elastic, _ = vem_lin_elast(
        g, C, bc, load=lambda x: np.zeros_like(x)
    )

    high_yield = solve_elastoplastic_vem(
        g, bc, young=E, poisson=nu, yield_stress=1.0e12,
        load_factors=[1.0],
    )
    assert np.allclose(high_yield.last.displacement, u_elastic, rtol=2e-7, atol=1e-10)
    assert not np.any(high_yield.last.plastic_state.yielded)
    assert high_yield.last.residual_norm < 1e-4
    print("  ok high-yield limit reproduces elastic VEM solve")

    low_yield = solve_elastoplastic_vem(
        g,
        bc,
        young=E,
        poisson=nu,
        yield_stress=6.0e4,
        hardening=1.0e7,
        load_factors=[0.25, 0.5, 0.75, 1.0],
        max_iter=16,
    )
    last = low_yield.last
    assert np.any(last.plastic_state.yielded)
    assert np.max(last.plastic_state.eq_plastic_strain) > 0.0
    assert last.residual_norm < 5e-3
    assert np.linalg.norm(last.displacement) > 0.0
    assert np.max(last.plastic_state.eq_plastic_strain) >= np.max(
        low_yield.steps[0].plastic_state.eq_plastic_strain
    )
    print(
        "  ok plastic load ramp: "
        f"steps={len(low_yield.steps)}, "
        f"yielded={np.count_nonzero(last.plastic_state.yielded)}, "
        f"max_eqp={np.max(last.plastic_state.eq_plastic_strain):.3e}"
    )
    print("Phase 102 validation passed: elastoplastic global coupling works.")


if __name__ == "__main__":
    main()
