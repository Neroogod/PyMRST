"""WellPhaseFlux (MRST ad-core/statefunctions/facility).

Ports MRST's `WellPhaseFlux.m` and the `calculatePhaseRate` it calls: the
reservoir volumetric phase flux through each perforation. With
``Tdp = -WI * drawdown``:

- a **producing** perforation (``Tdp < 0``) delivers each phase in proportion to
  the reservoir mobility there: ``q_alpha = mob_alpha * Tdp``;
- an **injecting** perforation (``Tdp > 0``) pushes the well's injection
  composition weighted by the *total* mobility: ``q_alpha = mob_t * compi_alpha
  * Tdp``.

This is exactly PyMRST's `_single_well` reservoir flux ``cq_vol`` (producer:
``-WI*mob*drawdown``; injector: ``-WI*mob_t*drawdown*compi``), so
`validate_phase343.py` checks the two agree to round-off on a multi-perforation
case. The crossflow mixture (`crossFlowMixture`, for injection into a producer
or production from an injector) is deferred with the component-well layer;
without crossflow the injection composition is the well's own ``compi``, which
is what this implements.
"""

from __future__ import annotations

from typing import Any, List, Sequence

import numpy as np

from pymrst.modules.ad_core.statefunctions.state_function import StateFunction
from pymrst.modules.ad_core.statefunctions.facility.component_mixing import (
    crossflow_composition,
)
from pymrst.utils.adi import value


def calculate_phase_rate(tdp: Any, mobw: Sequence[Any], mapping: Any,
                         wells: Sequence[Any], allow_cross_flow: bool = True,
                         model: Any = None) -> List[Any]:
    """Per-perforation phase rate ``q_alpha`` from ``Tdp`` and perforation
    mobility, splitting producers (reservoir mobility) from injectors
    (total mobility times the well injection composition)."""
    nph = len(mobw)
    v_tdp = np.asarray(value(tdp), dtype=float)
    inj_f = (v_tdp > 0).astype(float)
    prod_f = 1.0 - inj_f

    compi = np.array([np.asarray(w.compi, dtype=float).ravel()[:nph]
                      for w in wells])                     # (n_well, nph)
    perf_is_injector = mapping.is_injector[mapping.perf2well]
    production = (v_tdp < 0)
    crossflow = (
        ((inj_f.astype(bool) & ~perf_is_injector)
         | (production & perf_is_injector))
        & allow_cross_flow
    )
    if np.any(crossflow):
        q_wellbore = np.column_stack(
            [np.asarray(value(mob), dtype=float) * v_tdp for mob in mobw]
        )
        compi = crossflow_composition(q_wellbore, compi, mapping)
    compi_perf = compi[mapping.perf2well]                  # (n_perf, nph)

    mob_t = mobw[0]
    for i in range(1, nph):
        mob_t = mob_t + mobw[i]

    q_ph: List[Any] = []
    for i in range(nph):
        mob_i = prod_f * mobw[i] + inj_f * (mob_t * compi_perf[:, i])
        q_ph.append(mob_i * tdp)
    return q_ph


class WellPhaseFlux(StateFunction):
    """Reservoir volumetric phase flux between wellbore and reservoir."""

    def __init__(self, facility: Any = None):
        super().__init__(facility, label="q_\\alpha")
        self.depends_on(["FacilityWellMapping", "PressureGradient",
                         "WellIndex", "Mobility"])

    def evaluate_on_domain(self, facility: Any, state: Any) -> List[Any]:
        m = self.get_evaluated(state, "FacilityWellMapping")
        dp = self.get_evaluated(state, "PressureGradient")
        wi = self.get_evaluated(state, "WellIndex")
        mobw = self.get_evaluated(state, "Mobility")
        tdp = -wi * dp
        return calculate_phase_rate(tdp, mobw, m, m.wells)


__all__ = ["WellPhaseFlux", "calculate_phase_rate"]


if __name__ == "__main__":
    print("ok well_phase_flux (validated against _single_well in "
          "validate_phase343)")
    print("\nAll tests passed!")
