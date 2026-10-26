"""MRST ``ad-core/utils/pressureBCContribADI.m``: the surface flux of each
phase through pressure boundary faces, `-b mob T (p_bc - p_cell + g rho dz)`
with `dz` the cell centre's last coordinate less the face's, and `g` the
norm of MRST's global `gravity()`. `bc` must carry `cell2bcface`
(`bc2ADbc`). Returns `(bXqXbc, bc_cell)`, cells 0-based."""

from __future__ import annotations

from typing import Any, List, Sequence, Tuple

import numpy as np


def pressure_bc_contrib_adi(G: Any, s: Any, pX: Sequence[Any], rhoX: Sequence[Any],
                            mobX: Sequence[Any], bX: Sequence[Any], bc: Any) -> Tuple[List[Any], np.ndarray]:
    """MRST `[bXqXbc, bc_cell] = pressureBCContribADI(G, s, pX, rhoX, mobX, bX, bc)`."""
    from pymrst.params.gravity.gravity import gravity
    from pymrst.utils.adi import ad_matmul
    types = [str(t) for t in np.atleast_1d(_get(bc, "type"))]
    assert all(t == "pressure" for t in types), "only pressure bc allowed"
    face = np.asarray(_get(bc, "face"), dtype=np.int64).ravel()
    Tbc = np.asarray(_get(s, "T_all"), dtype=float)[face]
    N = np.asarray(G.faces.neighbors, dtype=np.int64)[face]
    assert np.all((N >= 0).sum(axis=1) == 1), "bc on internal boundary"
    c2f = _get(bc, "cell2bcface")
    bc_cell = N.max(axis=1)
    dzbc = np.asarray(G.cells.centroids)[bc_cell, -1] - np.asarray(G.faces.centroids)[face, -1]
    g = float(np.linalg.norm(gravity()))
    value_bc = np.asarray(_get(bc, "value"), dtype=float).ravel()
    out = []
    for i in range(len(pX)):
        pXbc = ad_matmul(c2f, pX[i])
        rhoXbc = ad_matmul(c2f, rhoX[i])
        dpbc = value_bc - pXbc + g*(rhoXbc*dzbc)
        bXmobXbc = ad_matmul(c2f, bX[i])*ad_matmul(c2f, mobX[i])
        out.append(-bXmobXbc*Tbc*dpbc)
    return out, bc_cell


def _get(obj: Any, name: str) -> Any:
    return obj.get(name) if isinstance(obj, dict) else getattr(obj, name, None)


pressureBCContribADI = pressure_bc_contrib_adi
__all__ = ["pressureBCContribADI", "pressure_bc_contrib_adi"]


if __name__ == "__main__":
    from types import SimpleNamespace

    from pymrst.gridprocessing import cart_grid, compute_geometry
    from pymrst.modules.ad_core.utils.bc_to_ad_bc import bc_to_ad_bc
    from pymrst.params.gravity.gravity import isolated_gravity
    from pymrst.params.wells_and_bc import add_bc

    def test_flux_follows_the_pressure_drop():
        G = compute_geometry(cart_grid([2, 1, 1], [2.0, 1.0, 1.0]))
        west = int(np.flatnonzero(np.isclose(G.faces.centroids[:, 0], 0.0))[0])
        bc = bc_to_ad_bc(G, add_bc(None, np.array([west]), "pressure", np.array([12.0])))
        s = SimpleNamespace(T_all=np.full(G.faces.num, 2.0))
        with isolated_gravity("off"):
            q, cells = pressure_bc_contrib_adi(G, s, [np.array([10.0, 8.0])], [np.ones(2)],
                                               [np.full(2, 3.0)], [np.ones(2)], bc)
        assert cells.tolist() == [0] and np.allclose(q[0], [-3*2*2.0])
        print("ok test_flux_follows_the_pressure_drop")

    test_flux_follows_the_pressure_drop()
    print("\nAll tests passed!")
