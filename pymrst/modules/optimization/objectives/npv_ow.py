"""Oil-water net-present-value objective (MRST ``NPVOW.m`` layout)."""

from typing import Optional, Sequence

import numpy as np

from pymrst.modules.optimization.adjoint import Objective
from pymrst.utils.adi import value


def npv_objective(
    oil_price: float = 60.0,
    water_cost: float = 6.0,
    water_inj_cost: float = 6.0,
    discount_rate: float = 0.0,
    dts: Optional[Sequence[float]] = None,
    time_unit: float = 365.0 * 86400.0,
) -> Objective:
    """Net present value for oil-water schedules."""
    if discount_rate and dts is None:
        raise ValueError("discounting needs the timestep sizes; pass dts=")
    elapsed = np.cumsum(np.asarray(dts, float)) if dts is not None else None

    def obj(model, p, sw, qws, qos, bhp, u, dt, n):
        j = None
        for k in range(len(model.wells)):
            qw, qo = qws[k:k + 1], qos[k:k + 1]
            producing = float(value(qo)[0]) + float(value(qw)[0]) < 0.0
            if producing:
                term = (-oil_price) * qo + water_cost * qw
            else:
                term = (-water_inj_cost) * qw
            j = term if j is None else j + term

        disc = (
            (1.0 + discount_rate) ** (-(elapsed[n] / time_unit))
            if discount_rate
            else 1.0
        )
        return j * (dt * disc)

    return obj


__all__ = ["npv_objective"]
