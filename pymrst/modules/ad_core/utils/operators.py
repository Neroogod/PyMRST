"""AD-core TPFA operators (MRST ``setupOperatorsTPFA`` layout)."""

from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np
import scipy.sparse as sp

from pymrst.gridprocessing.grid import Grid
from pymrst.params.rock import Rock
from pymrst.solvers.trans import compute_trans
from pymrst.utils.adi import ad_matmul


@dataclass
class Operators:
    """Discrete operators over interior grid connections."""

    N: np.ndarray
    C: sp.csr_array
    T: np.ndarray
    grad: Callable
    div: Callable
    avg: Callable
    upw: Callable
    pv: Optional[np.ndarray] = None
    t_all: Optional[np.ndarray] = None


def setup_operators_tpfa(
    g: Grid,
    rock: Optional[Rock] = None,
    t_half: Optional[np.ndarray] = None,
) -> Operators:
    """Assemble interior-connection operators with TPFA transmissibilities."""
    neighbors = g.faces.neighbors
    internal = np.all(neighbors >= 0, axis=1)
    n = neighbors[internal]
    n_if = n.shape[0]

    c = sp.csr_array(
        (
            np.tile([-1.0, 1.0], n_if),
            (np.repeat(np.arange(n_if), 2), n.ravel()),
        ),
        shape=(n_if, g.cells.num),
    )

    if t_half is None:
        assert rock is not None, "provide rock or half-transmissibilities"
        t_half = compute_trans(g, rock)
    with np.errstate(divide="ignore"):
        t_face = 1.0 / np.bincount(
            g.cells.faces,
            weights=1.0 / t_half,
            minlength=g.faces.num,
        )
    t = t_face[internal]

    c1, c2 = n[:, 0], n[:, 1]

    def grad(x):
        return ad_matmul(c, x)

    def div(x):
        return ad_matmul(-c.T, x)

    def avg(x):
        return 0.5 * (x[c1] + x[c2])

    def upw(flag: np.ndarray, x):
        m = sp.csr_array(
            (
                np.ones(n_if),
                (np.arange(n_if), np.where(flag, c1, c2)),
            ),
            shape=(n_if, g.cells.num),
        )
        return ad_matmul(m, x)

    pv = None
    if rock is not None and g.cells.volumes is not None:
        pv = rock.poro * g.cells.volumes

    return Operators(N=n, C=c, T=t, grad=grad, div=div, avg=avg, upw=upw, pv=pv, t_all=t_face)


__all__ = ["Operators", "setup_operators_tpfa"]
