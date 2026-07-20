"""Diagonal-backend TPFA operators.

These functions mirror the operator closures returned by
``setup_operators_tpfa`` but keep ``GenericAD`` Jacobians in compact diagonal
backend storage until an explicit sparse conversion is requested.
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp

from pymrst.modules.ad_core.backends.diagonal.generic_ad import GenericAD, diagonal_ad_matmul


def two_point_gradient(neighbors: np.ndarray, x):
    """Cell-to-face two-point difference ``x[c2] - x[c1]``."""
    n = np.asarray(neighbors, dtype=np.int64)
    return x[n[:, 1]] - x[n[:, 0]]


def discrete_divergence(neighbors: np.ndarray, ncells: int, face_values):
    """Face-to-cell divergence, equivalent to MRST/PyMRST ``-C.T``."""
    n = np.asarray(neighbors, dtype=np.int64)
    nfaces = n.shape[0]
    c = sp.csr_array(
        (np.tile([-1.0, 1.0], nfaces),
         (np.repeat(np.arange(nfaces), 2), n.ravel())),
        shape=(nfaces, int(ncells)),
    )
    return diagonal_ad_matmul(-c.T, face_values)


def face_average(neighbors: np.ndarray, x):
    """Arithmetic face average."""
    n = np.asarray(neighbors, dtype=np.int64)
    return 0.5 * (x[n[:, 0]] + x[n[:, 1]])


def single_point_upwind(flag: np.ndarray, neighbors: np.ndarray, x):
    """Single-point upwind: first neighbor where ``flag`` is true, else second."""
    n = np.asarray(neighbors, dtype=np.int64)
    flag = np.asarray(flag, dtype=bool).ravel()
    if flag.size != n.shape[0]:
        raise ValueError("upwind flag length must match number of interior faces")
    picked = np.where(flag, n[:, 0], n[:, 1])
    return x[picked]


def sparse_from_diagonal_ad(x: GenericAD):
    """Return ``(value, sparse_blocks)`` for solver handoff or diagnostics."""
    if not isinstance(x, GenericAD):
        return np.asarray(x, dtype=np.float64), []
    return x.val.copy(), [j.to_sparse() for j in x.jac]


def setup_diagonal_operator_closures(operators, ncells: int):
    """Patch an ``Operators`` object with diagonal-aware closures in-place."""
    neighbors = operators.N

    def grad(x):
        if isinstance(x, GenericAD):
            return two_point_gradient(neighbors, x)
        return operators.C @ x

    def div(x):
        if isinstance(x, GenericAD):
            return discrete_divergence(neighbors, ncells, x)
        return -operators.C.T @ x

    def avg(x):
        if isinstance(x, GenericAD):
            return face_average(neighbors, x)
        c1, c2 = neighbors[:, 0], neighbors[:, 1]
        return 0.5 * (x[c1] + x[c2])

    def upw(flag, x):
        if isinstance(x, GenericAD):
            return single_point_upwind(flag, neighbors, x)
        c1, c2 = neighbors[:, 0], neighbors[:, 1]
        return x[np.where(flag, c1, c2)]

    operators.grad = grad
    operators.div = div
    operators.avg = avg
    operators.upw = upw
    return operators


if __name__ == "__main__":
    from pymrst.gridprocessing import cart_grid, compute_geometry
    from pymrst.modules.ad_core.operators import setup_operators_tpfa
    from pymrst.modules.ad_core.backends.diagonal.generic_ad import init_variables_diagonal
    from pymrst.params.rock import make_rock
    from pymrst.utils.adi.init_variables_adi import init_variables_adi

    def test_diagonal_operators_match_sparse_adi():
        g = compute_geometry(cart_grid([3, 1, 1]))
        rock = make_rock(g, 1e-13, 0.2)
        ops = setup_operators_tpfa(g, rock)
        (pd,) = init_variables_diagonal(np.array([1.0, 4.0, 9.0]))
        (ps,) = init_variables_adi(np.array([1.0, 4.0, 9.0]))

        gd = two_point_gradient(ops.N, pd)
        gs = ops.grad(ps)
        assert np.allclose(gd.val, gs.val)
        assert np.allclose(gd.to_adi().jac[0].toarray(), gs.jac[0].toarray())

        dd = discrete_divergence(ops.N, g.cells.num, gd)
        ds = ops.div(gs)
        assert np.allclose(dd.val, ds.val)
        assert np.allclose(dd.to_adi().jac[0].toarray(), ds.jac[0].toarray())

        ad = face_average(ops.N, pd)
        assert np.allclose(ad.val, [2.5, 6.5])
        ud = single_point_upwind(np.array([True, False]), ops.N, pd)
        assert np.allclose(ud.val, [1.0, 9.0])
        assert np.allclose(ud.to_adi().jac[0].toarray(), [[1, 0, 0], [0, 0, 1]])
        print("ok test_diagonal_operators_match_sparse_adi")

    test_diagonal_operators_match_sparse_adi()
    print("\nAll tests passed!")
