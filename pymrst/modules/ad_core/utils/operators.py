"""Discrete TPFA operators on interior connections.

Ports MRST's ``ad-core/utils/setupOperatorsTPFA.m``.

Builds the sparse difference matrix C over interior faces and exposes
grad/div/avg closures that work on both NumPy arrays and ADI variables,
plus interior-face transmissibilities — everything an AD flow simulator
needs from the geological model.

Each closure dispatches on the operand: a ``GenericAD`` value routes to the
diagonal backend's compact kernels, anything else falls back to sparse
matmul.
"""

from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np
import scipy.sparse as sp

from pymrst.gridprocessing.grid import Grid
from pymrst.params.rock import Rock
from pymrst.solvers.trans import compute_trans
from pymrst.utils.adi import ad_matmul


def _is_generic_ad(x) -> bool:
    return x.__class__.__name__ == "GenericAD"


@dataclass
class Operators:
    """Discrete operators over the grid's interior connections."""

    N: np.ndarray                          # (n_if, 2) interior neighbors
    C: sp.csr_array                        # (n_if, nc) difference matrix
    T: np.ndarray                          # (n_if,) interior face trans
    grad: Callable                         # cell -> face difference
    div: Callable                          # face -> cell (adjoint of grad)
    avg: Callable                          # arithmetic face average
    upw: Callable                          # upwind face value given a flag
    pv: Optional[np.ndarray] = None        # reference pore volume
    # Transmissibility of *every* face, interior and boundary (MRST's T_all).
    # A boundary face has only one cell, so its value is that cell's
    # half-transmissibility - which is what a boundary condition flows through.
    t_all: Optional[np.ndarray] = None


def setup_operators_tpfa(g: Grid, rock: Optional[Rock] = None,
                         t_half: Optional[np.ndarray] = None) -> Operators:
    """Assemble interior-connection operators; T from rock or given t_half."""
    neighbors = g.faces.neighbors
    internal = np.all(neighbors >= 0, axis=1)
    n = neighbors[internal]
    nnc = getattr(g, "nnc", None)
    nnc_cells = np.zeros((0, 2), dtype=np.int64)
    nnc_trans = np.zeros(0, dtype=float)
    if nnc is not None:
        if isinstance(nnc, dict):
            nnc_cells = np.asarray(
                nnc.get("cells", np.zeros((0, 2))), dtype=np.int64
            ).reshape(-1, 2)
            nnc_trans = np.asarray(
                nnc.get("trans", np.zeros(nnc_cells.shape[0])), dtype=float
            ).ravel()
        else:
            nnc_cells = np.asarray(nnc.cells, dtype=np.int64).reshape(-1, 2)
            nnc_trans = np.asarray(nnc.trans, dtype=float).ravel()
        if nnc_trans.size != nnc_cells.shape[0]:
            raise ValueError("grid NNC cells/trans sizes do not agree")
        if np.any(nnc_cells < 0) or np.any(nnc_cells >= g.cells.num):
            raise IndexError("grid NNC references a cell outside the grid")
        n = np.vstack([n, nnc_cells])
    n_if = n.shape[0]

    c = sp.csr_array(
        (np.tile([-1.0, 1.0], n_if),
         (np.repeat(np.arange(n_if), 2), n.ravel())),
        shape=(n_if, g.cells.num),
    )

    if t_half is None:
        assert rock is not None, "provide rock or half-transmissibilities"
        t_half = compute_trans(g, rock)
    # Harmonic average of the half-transmissibilities. A sealing cell has
    # zero permeability and hence zero half-transmissibility, which the
    # harmonic mean handles correctly (1/0 -> inf -> t_face = 0, a sealed
    # face) - real decks contain such layers, so silence the warning rather
    # than let it spam a field-scale run. Norne's layer 4 is one.
    with np.errstate(divide="ignore"):
        t_face = 1.0 / np.bincount(g.cells.faces, weights=1.0 / t_half,
                                   minlength=g.faces.num)
    t = np.concatenate([t_face[internal], nnc_trans])

    c1, c2 = n[:, 0], n[:, 1]

    def grad(x):
        if _is_generic_ad(x):
            from pymrst.modules.ad_core.backends.diagonal.operators import two_point_gradient

            return two_point_gradient(n, x)
        return ad_matmul(c, x)

    def div(x):
        if _is_generic_ad(x):
            from pymrst.modules.ad_core.backends.diagonal.operators import discrete_divergence

            return discrete_divergence(n, g.cells.num, x)
        return ad_matmul(-c.T, x)

    def avg(x):
        if _is_generic_ad(x):
            from pymrst.modules.ad_core.backends.diagonal.operators import face_average

            return face_average(n, x)
        return 0.5 * (x[c1] + x[c2])

    def upw(flag: np.ndarray, x):
        """Upwind face value: x[c1] where flag else x[c2]."""
        if _is_generic_ad(x):
            from pymrst.modules.ad_core.backends.diagonal.operators import single_point_upwind

            return single_point_upwind(flag, n, x)
        m = sp.csr_array(
            (np.ones(n_if),
             (np.arange(n_if), np.where(flag, c1, c2))),
            shape=(n_if, g.cells.num),
        )
        return ad_matmul(m, x)

    pv = None
    if rock is not None and g.cells.volumes is not None:
        pv = rock.poro * g.cells.volumes

    return Operators(N=n, C=c, T=t, grad=grad, div=div, avg=avg, upw=upw,
                     pv=pv, t_all=t_face)


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
    print("\nAll tests passed!")
