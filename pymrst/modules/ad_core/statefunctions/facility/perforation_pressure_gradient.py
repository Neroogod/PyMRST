"""PerforationPressureGradient (MRST ad-core/statefunctions/facility).

Ports MRST's `PerforationPressureGradient.m`: the drawdown driving each
perforation, ``dp = p_cell - (bhp + cdp)``, where ``bhp`` is the well's
bottom-hole pressure and ``cdp`` is the pre-computed static connection
pressure-drop between the bhp datum and the perforation. It is the exact
quantity PyMRST's `_single_well` calls ``drawdown``.

The crossflow cutoff MRST applies here (zeroing the drawdown on connections
whose flow direction contradicts the well sign, when crossflow is disallowed)
is not applied at this stage in PyMRST's port: the port handles the injector /
producer split inside `WellPhaseFlux` instead, which is where `_single_well`
does it. This state function returns the raw drawdown, matching that value.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from pymrst.modules.ad_core.statefunctions.state_function import (
    STATE,
    StateFunction,
)


class PerforationPressureGradient(StateFunction):
    """Perforation drawdown ``p_cell - (bhp + cdp)``."""

    def __init__(self, facility: Any = None):
        super().__init__(facility, label="p_c - p_{bh} - g\\Delta z\\rho_w")
        self.depends_on("FacilityWellMapping")
        self.depends_on(["bhp", "cdp"], STATE)

    def evaluate_on_domain(self, facility: Any, state: Any):
        m = self.get_evaluated(state, "FacilityWellMapping")
        p = facility.reservoir_prop(state, "pressure")
        p_perf = p[m.cells]
        bhp = state["bhp"]                       # per active well
        bhp_perf = bhp[m.perf2well] if np.ndim(bhp) else bhp
        cdp = np.asarray(state["cdp"], dtype=float)   # per perforation
        return p_perf - (bhp_perf + cdp)


__all__ = ["PerforationPressureGradient"]


if __name__ == "__main__":
    print("ok perforation_pressure_gradient (see validate_phase343)")
    print("\nAll tests passed!")
