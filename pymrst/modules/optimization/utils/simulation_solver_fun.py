"""Forward simulation utility (MRST ``simulationSolverFun.m`` layout)."""

from typing import List, Sequence

import numpy as np

from pymrst.modules.ad_blackoil.models.two_phase_oil_water_model import (
    OWState,
    TwoPhaseOilWaterModelAD,
)


def _set_controls(model: TwoPhaseOilWaterModelAD, u: np.ndarray) -> None:
    for w, val in zip(model.wells, np.asarray(u, float).ravel()):
        w.val = float(val)


def simulate_forward(
    model: TwoPhaseOilWaterModelAD,
    state0: OWState,
    dts: Sequence[float],
    controls: np.ndarray,
) -> List[OWState]:
    """Run the schedule, keeping every state for adjoint evaluation."""
    states = [state0]
    for n, dt in enumerate(dts):
        _set_controls(model, controls[n])
        st, ok, _ = model.step(states[-1], float(dt))
        if not ok:
            raise RuntimeError(f"forward step {n} did not converge")
        states.append(st)
    return states


__all__ = ["simulate_forward"]
