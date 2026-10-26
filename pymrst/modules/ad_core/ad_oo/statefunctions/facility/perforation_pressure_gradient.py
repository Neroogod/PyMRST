"""MRST ``ad-core/statefunctions/facility/PerforationPressureGradient.m``.

Pressure difference between each perforated cell and the wellbore at that
connection: `p_cell - (bhp + cdp)`. A well that forbids crossflow has the
difference zeroed (value only) where it would reverse the well's direction.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from pymrst.modules.ad_core.statefunctions.state_function import (
    STATE,
    StateFunction,
)
from pymrst.utils.adi import value


class PerforationPressureGradient(StateFunction):
    """`p_c - p_bh - g dz rho_w` per perforation."""

    def __init__(self, model: Any):
        super().__init__(model, label="p_c-p_{bh}-g \\Delta z \\rho_{w}")
        self.AutoDiffBackend = getattr(model, "AutoDiffBackend", None)
        self.depends_on("FacilityWellMapping")
        self.depends_on("pressure", STATE)
        self.depends_on("bhp", STATE)

    def evaluate_on_domain(self, model: Any, state: Any) -> Any:
        mapping = self.get_evaluated_dependencies(state, "FacilityWellMapping")
        well_sol = state["wellSol"]
        p = model.ReservoirModel.get_props(state, "pressure")
        names = [n.lower() for n in state["FacilityState"]["names"]]
        active_rows = [well_sol[i] for i in mapping.active]
        if "bhp" in names:
            bhp = state["FacilityState"]["primaryVariables"][names.index("bhp")]
            bhp = bhp[mapping.perf2well]
        else:
            bhp = np.array([row.bhp for row in active_rows])[mapping.perf2well]
        # MATLAB's `vertcat` of no wells is empty.
        cdp = (np.concatenate([np.asarray(row.cdp, dtype=float).ravel() for row in active_rows])
               if active_rows else np.zeros(0))
        dp = p[mapping.cells] - (bhp + cdp)
        allow_xflow = np.array([model.WellModels[i].allowCrossflow
                                for i in mapping.active], dtype=bool)
        if not np.all(allow_xflow):
            vdp = np.asarray(value(dp), dtype=float)
            sgn = np.array([w.sign for w in mapping.W])
            perf_allow = allow_xflow[mapping.perf2well]
            psgn = sgn[mapping.perf2well]
            keep = perf_allow | ((psgn == 1) & (vdp <= 0)) | ((psgn == -1) & (vdp >= 0))
            if hasattr(dp, "jac"):
                dp = dp + 0.0
                dp.val[~keep] = 0.0
            else:
                dp = np.where(keep, dp, 0.0)
        return dp

    evaluateOnDomain = evaluate_on_domain


__all__ = ["PerforationPressureGradient"]


if __name__ == "__main__":
    from types import SimpleNamespace

    def _setup(allow: bool):
        mapping = SimpleNamespace(active=[0], perf2well=np.array([0, 0]),
                                  cells=np.array([0, 1]),
                                  W=[SimpleNamespace(sign=-1)])
        facility = SimpleNamespace(
            ReservoirModel=SimpleNamespace(get_props=lambda s, n: s["pressure"]),
            WellModels=[SimpleNamespace(allowCrossflow=allow)])
        state = {"wellSol": [SimpleNamespace(bhp=95.0, cdp=np.array([0.0, 1.0]))],
                 "pressure": np.array([100.0, 90.0]),
                 "FacilityState": {"names": [], "primaryVariables": []}}
        prop = PerforationPressureGradient(facility)
        prop.get_evaluated_dependencies = lambda st, name: mapping
        return prop, facility, state

    def test_drawdown_is_cell_pressure_less_connection_pressure():
        prop, facility, state = _setup(True)
        np.testing.assert_allclose(prop.evaluate_on_domain(facility, state),
                                   [5.0, -6.0])
        print("ok test_drawdown_is_cell_pressure_less_connection_pressure")

    def test_a_producer_without_crossflow_ignores_injecting_connections():
        prop, facility, state = _setup(False)
        np.testing.assert_allclose(prop.evaluate_on_domain(facility, state),
                                   [5.0, 0.0])
        print("ok test_a_producer_without_crossflow_ignores_injecting_connections")

    test_drawdown_is_cell_pressure_less_connection_pressure()
    test_a_producer_without_crossflow_ignores_injecting_connections()
    print("\nAll tests passed!")
