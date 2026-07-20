"""MRST ``twophaseJacobian.m`` two-phase transport residual/Jacobian."""

from typing import Optional, Sequence, Tuple

import numpy as np
import scipy.sparse as sp

from pymrst.gridprocessing.grid import Grid
from pymrst.modules.incomp.fluid import Fluid
from pymrst.modules.incomp.transport import gravity_flux, transport_source
from pymrst.params.gravity_const import gravity_or_zero
from pymrst.params.rock import Rock, pore_volume
from pymrst.params.wells_and_bc import BoundaryCondition, Source, Well
from pymrst.solvers.state import State


def _upwind_indices(n_int: np.ndarray, dflux: np.ndarray,
                    gflux: np.ndarray, mob: np.ndarray):
    iw = np.full(dflux.size, -1, dtype=np.int64)
    io = np.full(dflux.size, -1, dtype=np.int64)
    a = ~(dflux < 0) & ~(gflux < 0)
    b = ~(dflux > 0) & ~(gflux > 0)
    c = a | b
    n = n_int.copy()
    n[b] = n[b][:, ::-1]
    iw[c] = n[c, 0]
    a = ~(dflux < 0) & ~(gflux > 0)
    b = ~(dflux > 0) & ~(gflux < 0)
    c = a | b
    n = n_int.copy()
    n[b] = n[b][:, ::-1]
    io[c] = n[c, 0]
    undecided = (iw < 0) | (io < 0)
    if np.any(undecided):
        mw = np.where(iw >= 0, mob[np.maximum(iw, 0), 0], np.nan)
        mo = np.where(io >= 0, mob[np.maximum(io, 0), 1], np.nan)
        vw = dflux + gflux * mo
        vo = dflux - gflux * mw
        n = n_int.copy()
        n[vw < 0] = n[vw < 0][:, ::-1]
        iw[iw < 0] = n[iw < 0, 0]
        n = n_int.copy()
        n[vo < 0] = n[vo < 0][:, ::-1]
        io[io < 0] = n[io < 0, 0]
    return iw, io


def two_phase_residual_jacobian(
    g: Grid,
    state: State,
    rock: Rock,
    fluid: Fluid,
    wells: Optional[Sequence[Well]] = None,
    src: Optional[Source] = None,
    bc: Optional[BoundaryCondition] = None,
    gravity: Optional[np.ndarray] = None,
) -> Tuple:
    """Build residual ``F(s,s0,dt)`` and Jacobian ``dF/ds``."""
    assert fluid.n_phases == 2, "two-phase transport requires a 2-phase fluid"
    q = transport_source(state, g, fluid, wells=wells, src=src, bc=bc)
    g_vec = gravity_or_zero(gravity, g.griddim)
    gflux = gravity_flux(g, rock, g_vec * (fluid.rho[0] - fluid.rho[1]))

    neighbors = g.faces.neighbors
    internal = np.all(neighbors >= 0, axis=1)
    n_int = neighbors[internal]
    ic1, ic2 = n_int[:, 0], n_int[:, 1]
    dflux = state.flux[internal]
    gflux_int = gflux[internal]
    pv = pore_volume(g, rock)
    nc = g.cells.num

    def upwind(mob: np.ndarray):
        return _upwind_indices(n_int, dflux, gflux_int, mob)

    def residual(s: np.ndarray, s0: np.ndarray, dt: float) -> np.ndarray:
        sat = np.column_stack([s, 1.0 - s])
        m = fluid.mobility(sat)
        f_cell = m[:, 0] / m.sum(axis=1)
        q_w = np.maximum(q, 0.0) + np.minimum(q, 0.0) * f_cell
        iw, io = upwind(m)
        mw, mo = m[iw, 0], m[io, 1]
        f_face = mw / (mw + mo)
        v_water = f_face * (dflux + gflux_int * mo)
        div = np.bincount(np.concatenate([ic1, ic2]),
                          weights=np.concatenate([v_water, -v_water]),
                          minlength=nc)
        return s - s0 + dt / pv * (div - q_w)

    def jacobian(s: np.ndarray, s0: np.ndarray, dt: float) -> sp.csr_array:
        sat = np.column_stack([s, 1.0 - s])
        m = fluid.mobility(sat)
        dm = fluid.d_mobility(sat)
        dm = np.column_stack([dm[:, 0], -dm[:, 1]])

        iw, io = upwind(m)
        mw, mo = m[iw, 0], m[io, 1]
        mt = mw + mo
        f_face = np.column_stack([mw / mt, mo / mt])

        dmw_fo = f_face[:, 1] * dm[iw, 0]
        dmo_fw = f_face[:, 0] * dm[io, 1]
        dvw_ds1 = (dflux + gflux_int * mo) * dmw_fo / mt
        dvw_ds2 = -(dflux - gflux_int * mw) * dmo_fw / mt

        mt = m.sum(axis=1)
        df = (m[:, 1] * dm[:, 0] - m[:, 0] * dm[:, 1]) / mt**2
        d = 1.0 - dt / pv * (np.minimum(q, 0.0) * df)

        cid = np.arange(nc)
        rows = np.concatenate([cid, ic1, ic2, ic1, ic2])
        cols = np.concatenate([cid, iw, iw, io, io])
        vals = np.concatenate([
            d,
            dt / pv[ic1] * dvw_ds1, -(dt / pv[ic2] * dvw_ds1),
            dt / pv[ic1] * dvw_ds2, -(dt / pv[ic2] * dvw_ds2),
        ])
        return sp.coo_array((vals, (rows, cols)), shape=(nc, nc)).tocsr()

    return residual, jacobian, gflux, q

__all__ = ["two_phase_residual_jacobian"]
