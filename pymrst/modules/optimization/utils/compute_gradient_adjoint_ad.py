"""Adjoint gradient utility (MRST ``computeGradientAdjointAD`` layout)."""

from typing import Optional, Sequence

import numpy as np
import scipy.sparse.linalg as spla

from pymrst.modules.ad_blackoil.models.two_phase_oil_water_model import (
    OWState,
    TwoPhaseOilWaterModelAD,
)
from pymrst.modules.optimization.adjoint import (
    AdjointResult,
    Objective,
    _jac_wrt_controls,
    _jac_wrt_current,
    _jac_wrt_previous,
    _objective_derivatives,
    _set_controls,
)
from pymrst.modules.optimization.utils.simulation_solver_fun import simulate_forward


def compute_gradient_adjoint(
    model: TwoPhaseOilWaterModelAD,
    state0: OWState,
    dts: Sequence[float],
    controls: np.ndarray,
    objective: Objective,
    states: Optional[Sequence[OWState]] = None,
) -> AdjointResult:
    """Compute ``dJ/du`` for each control of each step by a backward sweep."""
    controls = np.atleast_2d(np.asarray(controls, float))
    nsteps, nw = controls.shape
    assert len(dts) == nsteps, "one control row per timestep"
    assert nw == len(model.wells), "one control per well"

    if states is None:
        states = simulate_forward(model, state0, dts, controls)

    dRdu = _jac_wrt_controls(model)
    grad = np.zeros((nsteps, nw))
    jtotal = 0.0
    lam = None

    for n in reversed(range(nsteps)):
        _set_controls(model, controls[n])
        st_prev, st = states[n], states[n + 1]
        dt = float(dts[n])

        jn, djdx = _objective_derivatives(model, objective, st, controls[n], dt, n)
        jtotal += jn

        rhs = -djdx
        if lam is not None:
            _set_controls(model, controls[n + 1])
            b = _jac_wrt_previous(model, st, states[n + 2], float(dts[n + 1]))
            rhs = rhs - b.T @ lam
            _set_controls(model, controls[n])

        a = _jac_wrt_current(model, st_prev, st, dt)
        lam = spla.spsolve(a.T.tocsc(), rhs)
        grad[n] = dRdu.T @ lam

    return AdjointResult(value=jtotal, gradient=grad, states=list(states))


__all__ = ["AdjointResult", "compute_gradient_adjoint"]
