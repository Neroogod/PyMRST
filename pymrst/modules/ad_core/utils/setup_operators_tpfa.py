"""MRST ``ad-core/utils/setupOperatorsTPFA.m``: the discrete operators and
derived quantities of a two-point flux discretization.

`op = setup_operators_tpfa(G, rock, ...)` holds, under MRST's field names:
`T_all` (one transmissibility per interface, boundary half-transmissibilities
included), `T` (internal interfaces only), `N` (the internal neighbourship,
grid NNCs appended), `internalConn`, `pv` (the `porv` option, else the grid's
`PORV`, else `poreVolume`), the transfer matrix `C` (+1 at a connection's
first cell, -1 at its second), `Grad = -C x`, `Div = C' x`,
`AccDiv(acc, flux) = acc + Div(flux)`, the averaging matrix `M` and
`faceAvg = M x`, and `faceUpstr(flag, x)`.

Transmissibilities come from the grid (`G.faces.TRANS`) or from
`getFaceTransmissibility(G, rock)`, or are given: `trans` alone (per internal
interface or per face) or with `neighbors` (MRST's explicit cell pairs, as an
upscaled model passes them). Indices are 0-based; outside is -1.

Two PyMRST additions. Each operator, given a diagonal-backend AD value, uses
the diagonal backend's own kernels, which is where MRST's
`DiagonalAutoDiffBackend.updateDiscreteOperators` puts them; the numbers are
the same. And the operators also answer to the names the rest of this port
uses: `grad`, `div`, `avg`, `upw`, `t_all`.
"""

from __future__ import annotations

import warnings
from typing import Any, Callable, Optional, Tuple

import numpy as np
import scipy.sparse as sp

from pymrst.utils.adi import ad_matmul


class Operators:
    """MRST's operators struct."""

    def __init__(self, **fields: Any) -> None:
        self.gdz = None
        self.__dict__.update(fields)

    # The names this port's models use for MRST's fields.
    @property
    def grad(self) -> Callable:
        return self.Grad

    @property
    def div(self) -> Callable:
        return self.Div

    @property
    def avg(self) -> Callable:
        return self.faceAvg

    @property
    def upw(self) -> Callable:
        return self.faceUpstr

    @property
    def t_all(self) -> np.ndarray:
        return self.T_all

    @t_all.setter
    def t_all(self, value: np.ndarray) -> None:
        self.T_all = value

    def get(self, name: str, default: Any = None) -> Any:
        """MATLAB's `isfield`-guarded field access."""
        return getattr(self, name, default)


def setup_operators_tpfa(G: Any, rock: Any = None, t_half: Optional[np.ndarray] = None, *,
                         deck: Any = None, neighbors: Optional[np.ndarray] = None,
                         trans: Optional[np.ndarray] = None,
                         porv: Optional[np.ndarray] = None) -> Operators:
    """MRST `op = setupOperatorsTPFA(G, rock, ...)`. `t_half`, PyMRST's
    earlier argument, gives one-sided transmissibilities to reduce instead of
    computing them from `rock`."""
    if deck is not None:
        warnings.warn("The 'deck' option is deprecated and inoperative. It will be removed "
                      "in a future version of MRST.")
    if trans is None:
        N, T, T_all, intInx = _grid_based_trans(G, rock, t_half, neighbors)
    else:
        N, T, T_all, intInx = _user_provided_trans(G, neighbors, trans)
    if _has_grid_nncs(G):
        N, T, T_all, intInx = _add_grid_nncs(N, T, T_all, intInx, G)
    if np.any(T < 0):
        _warn_negative_trans(T)
    op = Operators(T=T, T_all=T_all, N=N, internalConn=intInx, pv=_pore_volume(G, rock, porv))
    nc, nf = _add_div_grad_operators(op, G, N)
    _add_face_average_operators(op, N, nf, nc)
    op.faceUpstr = lambda flag, x: _face_upstr(flag, x, N, (nf, nc))
    return op


def _grid_based_trans(G: Any, rock: Any, t_half: Optional[np.ndarray],
                      neighbors: Optional[np.ndarray]) -> Tuple[np.ndarray, ...]:
    from pymrst.solvers.get_face_transmissibility import get_face_transmissibility
    assert neighbors is None
    grid_trans = getattr(G.faces, "TRANS", None)
    if grid_trans is not None:
        T = np.asarray(grid_trans, dtype=float).ravel()
    else:
        with np.errstate(divide="ignore"):
            T = get_face_transmissibility(G, rock, t_half)
    N = np.asarray(G.faces.neighbors, dtype=np.int64)
    intInx = np.all(N >= 0, axis=1)
    return N[intInx], T[intInx], T, intInx


def _user_provided_trans(G: Any, N: Optional[np.ndarray], T: np.ndarray) -> Tuple[np.ndarray, ...]:
    T = np.asarray(T, dtype=float).ravel()
    if N is None:
        N, intInx, n_if = _internal_grid_neighbours(G)
    else:
        N, intInx, n_if = _explicit_cell_pairs(np.asarray(N, dtype=np.int64).reshape(-1, 2), G)
    if T.size == n_if:
        T_all = np.zeros(intInx.size)
        T_all[intInx] = T
    else:
        assert T.size == intInx.size
        T_all = T
        T = T[intInx]
    return N, T, T_all, intInx


def _add_grid_nncs(N, T, T_all, intInx, G) -> Tuple[np.ndarray, ...]:
    cells = np.asarray(_nnc_field(G, "cells"), dtype=np.int64).reshape(-1, 2)
    trans = np.asarray(_nnc_field(G, "trans"), dtype=float).ravel()
    known = {tuple(r) for r in np.sort(N, axis=1)}
    present = np.array([tuple(r) in known for r in np.sort(cells, axis=1)], dtype=bool)
    i = ~(np.any(cells < 0, axis=1) | present)
    return (np.vstack([N, cells[i]]), np.concatenate([T, trans[i]]),
            np.concatenate([T_all, trans[i]]),
            np.concatenate([intInx, np.ones(int(i.sum()), dtype=bool)]))


def _pore_volume(G: Any, rock: Any, porv: Optional[np.ndarray]) -> Optional[np.ndarray]:
    if porv is not None:
        return np.asarray(porv, dtype=float).ravel()
    grid_porv = getattr(G.cells, "PORV", None)
    if grid_porv is not None:
        pv = np.asarray(grid_porv, dtype=float).ravel()
    elif rock is not None and getattr(G.cells, "volumes", None) is not None:
        from pymrst.params.rock.pore_volume import pore_volume
        pv = np.asarray(pore_volume(G, rock), dtype=float).ravel()
    else:
        return None
    numzeropv = int(np.count_nonzero(~(pv > 0)))
    if numzeropv > 0:
        warnings.warn(f"I computed zero pore volumes in {numzeropv} cells.  Consider adjusting "
                      "'poro'/'ntg' fields or the grid itself.")
    return pv


def _add_div_grad_operators(op: Operators, G: Any, N: np.ndarray) -> Tuple[int, int]:
    nc = G.cells.num if op.pv is None else op.pv.size
    nf = N.shape[0]
    assert nc == G.cells.num, "Dimension mismatch between grid and supplied pore-volumes."
    C = sp.csr_array((np.tile([1.0, -1.0], nf), (np.repeat(np.arange(nf), 2), N.ravel())),
                     shape=(nf, nc))
    Ct = C.T.tocsr()
    op.C = C

    def Grad(x):
        if _is_generic_ad(x):
            from pymrst.modules.ad_core.backends.diagonal.operators import two_point_gradient
            return two_point_gradient(N, x)
        return -ad_matmul(C, x)

    def Div(x):
        if nf == 0:
            return np.zeros(nc)
        if _is_generic_ad(x):
            from pymrst.modules.ad_core.backends.diagonal.operators import discrete_divergence
            return discrete_divergence(N, nc, x)
        return ad_matmul(Ct, x)

    op.Grad = Grad
    op.Div = Div
    op.AccDiv = (lambda acc, flux: acc) if nf == 0 else (lambda acc, flux: acc + Div(flux))
    return nc, nf


def _add_face_average_operators(op: Operators, N: np.ndarray, nf: int, nc: int) -> None:
    M = sp.csr_array((np.full(2*nf, 0.5), (np.repeat(np.arange(nf), 2), N.ravel())),
                     shape=(nf, nc))
    op.M = M

    def faceAvg(x):
        if _is_generic_ad(x):
            from pymrst.modules.ad_core.backends.diagonal.operators import face_average
            return face_average(N, x)
        return ad_matmul(M, x)

    op.faceAvg = faceAvg


def _face_upstr(flag: Any, x: Any, N: np.ndarray, sz: Tuple[int, int]) -> Any:
    if _is_generic_ad(x):
        from pymrst.modules.ad_core.backends.diagonal.operators import single_point_upwind
        flag = np.asarray(flag, dtype=bool)
        if flag.size == 1:
            flag = np.repeat(flag.ravel(), N.shape[0])
        return single_point_upwind(flag, N, x)
    from pymrst.modules.ad_core.utils.face_upstr import face_upstr
    return face_upstr(flag, x, N, sz)


def _warn_negative_trans(T: np.ndarray) -> None:
    nneg = int(np.sum(T < 0))
    warnings.warn(f"Negative transmissibilities detected for {nneg}/{T.size} "
                  f"({100*nneg/T.size:.2e} %) connections.")


def _has_grid_nncs(G: Any) -> bool:
    nnc = getattr(G, "nnc", None)
    return nnc is not None and _nnc_field(G, "cells") is not None \
        and _nnc_field(G, "trans") is not None


def _nnc_field(G: Any, name: str) -> Any:
    nnc = G.nnc
    return nnc.get(name) if isinstance(nnc, dict) else getattr(nnc, name, None)


def _internal_grid_neighbours(G: Any) -> Tuple[np.ndarray, np.ndarray, int]:
    N = np.asarray(G.faces.neighbors, dtype=np.int64)
    intInx = np.all(N >= 0, axis=1)
    return N[intInx], intInx, int(intInx.sum())


def _explicit_cell_pairs(N: np.ndarray, G: Any) -> Tuple[np.ndarray, np.ndarray, int]:
    intInx = np.all(N >= 0, axis=1)
    n_if = int(intInx.sum())
    if getattr(G, "faces", None) is not None:
        intInxGrid = np.all(np.asarray(G.faces.neighbors) >= 0, axis=1)
        if intInxGrid.sum() == n_if:
            intInx = intInxGrid
    return N, intInx, n_if


def _is_generic_ad(x: Any) -> bool:
    return x.__class__.__name__ == "GenericAD"


setupOperatorsTPFA = setup_operators_tpfa
__all__ = ["Operators", "setupOperatorsTPFA", "setup_operators_tpfa"]


if __name__ == "__main__":
    from pymrst.gridprocessing import cart_grid, compute_geometry
    from pymrst.params.rock import make_rock
    from pymrst.utils.adi import init_variables_adi, value

    def test_grad_div_adjoint():
        """div is the negative adjoint of grad: <grad u, v> = <u, div v>."""
        g = compute_geometry(cart_grid([4, 3, 2]))
        rock = make_rock(g, 1e-13, 0.2)
        ops = setup_operators_tpfa(g, rock)
        rng = np.random.default_rng(0)
        u = rng.random(g.cells.num)
        v = rng.random(ops.N.shape[0])
        assert np.isclose(ops.grad(u) @ v, -(u @ ops.div(v)))
        print("ok test_grad_div_adjoint")

    def test_grad_of_linear_field():
        """grad of x-linear cell field equals dx on x-connections."""
        g = compute_geometry(cart_grid([5, 1, 1], [50.0, 1.0, 1.0]))
        rock = make_rock(g, 1e-13, 0.2)
        ops = setup_operators_tpfa(g, rock)
        gx = ops.grad(g.cells.centroids[:, 0])
        assert np.allclose(gx, 10.0)
        print("ok test_grad_of_linear_field")

    def test_operators_propagate_adi():
        """grad/div/avg/upw all carry Jacobians through."""
        g = compute_geometry(cart_grid([3, 1, 1]))
        rock = make_rock(g, 1e-13, 0.2)
        ops = setup_operators_tpfa(g, rock)
        (p,) = init_variables_adi(np.array([1.0, 4.0, 9.0]))
        gp = ops.grad(p)
        assert np.allclose(gp.val, [3.0, 5.0])
        assert np.allclose(gp.jac[0].toarray(),
                           [[-1, 1, 0], [0, -1, 1]])
        d = ops.div(gp)   # div(grad p): Jacobian is -C'C (negative Laplacian)
        assert np.allclose(d.jac[0].toarray(),
                           [[-1, 1, 0], [1, -2, 1], [0, 1, -1]])
        a = ops.avg(p)
        assert np.allclose(a.val, [2.5, 6.5])
        u = ops.upw(np.array([True, False]), p)
        assert np.allclose(value(u), [1.0, 9.0])
        assert np.allclose(u.jac[0].toarray(), [[1, 0, 0], [0, 0, 1]])
        print("ok test_operators_propagate_adi")

    def test_face_transmissibility():
        """Interior T equals harmonic mean of the two half-face values."""
        g = compute_geometry(cart_grid([2, 1, 1], [2.0, 1.0, 1.0]))
        rock = make_rock(g, np.array([[1.0], [3.0]]), 0.2)
        ops = setup_operators_tpfa(g, rock)
        t1, t2 = 1.0 / 0.5, 3.0 / 0.5   # K*A/(d/2)
        assert np.allclose(ops.T, 1.0 / (1.0 / t1 + 1.0 / t2))
        print("ok test_face_transmissibility")

    test_grad_div_adjoint()
    test_grad_of_linear_field()
    test_operators_propagate_adi()
    test_face_transmissibility()

    def test_pore_volume_follows_mrsts_order():
        from pymrst.gridprocessing import cart_grid, compute_geometry
        from pymrst.params.rock import make_rock
        g = compute_geometry(cart_grid([2, 1, 1]))
        rock = make_rock(g, 1e-13, 0.2, ntg=0.5)
        np.testing.assert_allclose(setup_operators_tpfa(g, rock).pv, [0.1, 0.1])
        g.cells.PORV = np.array([3.0, 4.0])
        np.testing.assert_allclose(setup_operators_tpfa(g, rock).pv, [3.0, 4.0])
        np.testing.assert_allclose(setup_operators_tpfa(g, rock, porv=np.ones(2)).pv, [1, 1])
        print("ok test_pore_volume_follows_mrsts_order")

    test_pore_volume_follows_mrsts_order()
    def test_mrst_fields_and_sign_convention():
        """MRST's C is +1 at a connection's first cell and -1 at its second;
        Grad = -C x, Div = C' x, M averages, T_all keeps boundary faces."""
        g = compute_geometry(cart_grid([3, 1, 1]))
        ops = setup_operators_tpfa(g, make_rock(g, 1e-13, 0.2))
        assert np.array_equal(ops.C.toarray(), [[1, -1, 0], [0, 1, -1]])
        assert np.array_equal(ops.M.toarray(), [[0.5, 0.5, 0], [0, 0.5, 0.5]])
        assert ops.T_all.size == g.faces.num and ops.internalConn.sum() == 2
        assert ops.t_all is ops.T_all and ops.get("missing", 7) == 7
        assert np.allclose(ops.Div(np.array([1.0, 2.0])), [1.0, 1.0, -2.0])
        assert np.allclose(ops.AccDiv(np.ones(3), np.array([1.0, 2.0])), [2.0, 2.0, -1.0])
        assert np.allclose(ops.faceUpstr(True, np.array([1.0, 2.0, 3.0])), [1.0, 2.0])
        print("ok test_mrst_fields_and_sign_convention")

    def test_given_transmissibilities_and_grid_nncs():
        g = compute_geometry(cart_grid([3, 1, 1]))
        ops = setup_operators_tpfa(g, trans=np.array([2.0, 3.0]))
        assert np.allclose(ops.T, [2.0, 3.0]) and np.allclose(ops.T_all[ops.internalConn], [2, 3])
        g.nnc = {"cells": np.array([[0, 2], [1, 0]]), "trans": np.array([5.0, 9.0])}
        ops = setup_operators_tpfa(g, make_rock(g, 1e-13, 0.2))
        # (0, 2) is new; (1, 0) is the existing connection (0, 1) and is skipped.
        assert ops.N.shape == (3, 2) and np.array_equal(ops.N[-1], [0, 2]) and ops.T[-1] == 5.0
        print("ok test_given_transmissibilities_and_grid_nncs")

    def test_grid_trans_and_negative_warning():
        import warnings
        g = compute_geometry(cart_grid([3, 1, 1]))
        g.faces.TRANS = -np.ones(g.faces.num)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            ops = setup_operators_tpfa(g, make_rock(g, 1e-13, 0.2))
        assert np.allclose(ops.T, -1.0) and any("Negative" in str(w.message) for w in caught)
        print("ok test_grid_trans_and_negative_warning")

    test_mrst_fields_and_sign_convention()
    test_given_transmissibilities_and_grid_nncs()
    test_grid_trans_and_negative_warning()
    print("\nAll tests passed!")
