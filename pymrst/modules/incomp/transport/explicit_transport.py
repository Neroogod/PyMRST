"""MRST ``explicitTransport.m`` forward-Euler two-phase transport."""

from typing import Optional, Sequence

import numpy as np

from pymrst.gridprocessing.grid import Grid
from pymrst.modules.incomp.fluid import Fluid
from pymrst.modules.incomp.transport import (
    _estimate_dt,
    gravity_flux,
    two_phase_residual_jacobian,
)
from pymrst.params.gravity_const import gravity_or_zero
from pymrst.params.rock import Rock
from pymrst.params.wells_and_bc import BoundaryCondition, Source, Well
from pymrst.solvers.state import State


def explicit_transport(
    state: State,
    g: Grid,
    tf: float,
    rock: Rock,
    fluid: Fluid,
    wells: Optional[Sequence[Well]] = None,
    src: Optional[Source] = None,
    bc: Optional[BoundaryCondition] = None,
    gravity: Optional[np.ndarray] = None,
    dt: Optional[float] = None,
    max_dt: float = np.inf,
    dt_factor: float = 0.5,
    satwarn: float = np.sqrt(np.finfo(float).eps),
) -> State:
    """Advance saturations ``tf`` seconds with forward-Euler upwind steps."""
    residual, _, _, q = two_phase_residual_jacobian(
        g, state, rock, fluid, wells=wells, src=src, bc=bc, gravity=gravity
    )
    g_vec = gravity_or_zero(gravity, g.griddim)
    gflux_est = gravity_flux(g, rock, -g_vec)

    s = state.s[:, 0].copy()
    t = 0.0
    while t < tf:
        if dt is None:
            step = min(
                max_dt,
                dt_factor * _estimate_dt(g, state, rock, fluid, gflux_est, q),
            )
        else:
            step = dt
        step = min(tf - t, step)
        s = s - residual(s, s, step)
        t += step

        assert not np.any(s > 1 + satwarn) and not np.any(s < -satwarn), (
            "explicit_transport: saturation outside [0, 1] beyond satwarn"
        )
        s = np.clip(s, 0.0, 1.0)
        state.s = np.column_stack([s, 1.0 - s])

    assert not np.any(np.isnan(s)), "explicit_transport: transport failed"
    return state

__all__ = ["explicit_transport"]
