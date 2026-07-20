"""MRST ``implicitTransport.m`` implicit two-phase transport."""

from typing import Optional, Sequence

import numpy as np
import scipy.sparse.linalg as spla

from pymrst.gridprocessing.grid import Grid
from pymrst.modules.incomp.fluid import Fluid
from pymrst.modules.incomp.transport import _linesearch, two_phase_residual_jacobian
from pymrst.params.rock import Rock
from pymrst.params.wells_and_bc import BoundaryCondition, Source, Well
from pymrst.solvers.state import State


def implicit_transport(
    state: State,
    g: Grid,
    tf: float,
    rock: Rock,
    fluid: Fluid,
    wells: Optional[Sequence[Well]] = None,
    src: Optional[Source] = None,
    bc: Optional[BoundaryCondition] = None,
    gravity: Optional[np.ndarray] = None,
    nltol: float = 1e-6,
    lstrials: int = 20,
    maxnewt: int = 25,
    tsref: int = 12,
    resred: float = 0.99,
) -> State:
    """Advance saturations ``tf`` seconds with backward Euler and Newton."""
    residual, jacobian, _, _ = two_phase_residual_jacobian(
        g, state, rock, fluid, wells=wells, src=src, bc=bc, gravity=gravity
    )

    s = state.s[:, 0].copy()
    mints = tf * 2.0 ** (-tsref)
    t, dt, dtprev, count = 0.0, tf, -tf, 0

    while t < tf and dt >= mints:
        dt = min(dt, tf - t)
        while True:
            s0 = s.copy()
            sn = np.minimum(1.0, s + 0.05)
            res = residual(sn, s0, dt)
            err = np.abs(res).max()
            it = 0
            linfail = False
            while err > nltol and not linfail and it < maxnewt:
                jac = jacobian(sn, s0, dt)
                ds = -spla.spsolve(jac.tocsc(), res)
                sn, res, linfail = _linesearch(
                    sn, ds, resred * err,
                    lambda x: residual(x, s0, dt), lstrials
                )
                it += 1
                err = np.abs(res).max()

            count -= 1
            if err > nltol:
                dt = dtprev if (count > 0 and dtprev > 0) else dt / 2.0
                count = 5
            else:
                t += dt
                dtprev = -dt
                if t < tf and it <= 5 and count < 0:
                    dtprev = dt
                    dt = min(1.5 * dt, tf - t)
                    count = 5
                break
        s = sn

    assert t >= tf and not np.any(np.isnan(s)), (
        f"implicit_transport failed: t={t:.3e} of {tf:.3e}, "
        f"minimum substep {mints:.3e} reached"
    )
    state.s = np.column_stack([s, 1.0 - s])
    return state

__all__ = ["implicit_transport"]
