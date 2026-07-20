"""EDFM TPFA pressure operator (MRST ``setupEDFMOperatorsTPFA.m`` layout)."""

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from pymrst.modules.hfm.edfm2d import EDFMGrid


def solve_incomp_edfm(
    e: EDFMGrid,
    mu: float,
    bc_cells: np.ndarray,
    bc_trans: np.ndarray,
    bc_press: np.ndarray,
) -> np.ndarray:
    """Solve incompressible pressure over matrix plus fracture cells."""
    n = e.n_total
    t = e.trans / mu
    c1, c2 = e.conn[:, 0], e.conn[:, 1]
    d = np.bincount(
        np.concatenate([c1, c2]),
        weights=np.concatenate([t, t]),
        minlength=n,
    )
    rhs = np.zeros(n)
    tb = bc_trans / mu
    np.add.at(d, bc_cells, tb)
    np.add.at(rhs, bc_cells, tb * bc_press)
    rows = np.concatenate([c1, c2, np.arange(n)])
    cols = np.concatenate([c2, c1, np.arange(n)])
    vals = np.concatenate([-t, -t, d])
    a = sp.csr_array((vals, (rows, cols)), shape=(n, n))
    return spla.spsolve(a.tocsc(), rhs)


def setup_edfm_operators_tpfa(
    e: EDFMGrid,
    mu: float,
    bc_cells: np.ndarray,
    bc_trans: np.ndarray,
    bc_press: np.ndarray,
) -> np.ndarray:
    """MRST-shaped entrypoint returning the EDFM pressure solution."""
    return solve_incomp_edfm(e, mu, bc_cells, bc_trans, bc_press)


__all__ = ["EDFMGrid", "solve_incomp_edfm", "setup_edfm_operators_tpfa"]
