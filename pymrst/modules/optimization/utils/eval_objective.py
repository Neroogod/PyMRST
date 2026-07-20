"""Objective evaluation utility (MRST ``evalObjective.m`` layout)."""

from typing import Sequence

import numpy as np

from pymrst.modules.ad_blackoil.models.two_phase_oil_water_model import (
    OWState,
    TwoPhaseOilWaterModelAD,
)
from pymrst.modules.optimization.adjoint import Objective
from pymrst.utils.adi import value


def evaluate_objective(
    model: TwoPhaseOilWaterModelAD,
    objective: Objective,
    states: Sequence[OWState],
    dts: Sequence[float],
    controls: np.ndarray,
) -> float:
    """Evaluate ``J = sum_n J_n`` on an already-simulated schedule."""
    total = 0.0
    for n, dt in enumerate(dts):
        j = objective(
            model,
            states[n + 1].pressure,
            states[n + 1].sW,
            states[n + 1].qWs,
            states[n + 1].qOs,
            states[n + 1].bhp,
            controls[n],
            float(dt),
            n,
        )
        total += float(np.asarray(value(j)).ravel()[0])
    return total


__all__ = ["evaluate_objective"]
