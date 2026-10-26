"""MRST ``ad-core/statefunctions/facility/WellPhaseFlux.m``.

Phase flux between each perforated cell and the wellbore,
`q_alpha = lambda_alpha * (-WI dp)`, with injection carried at the total
mobility in the well's (crossflow-corrected) injection composition.
"""

from __future__ import annotations

from typing import Any, List

from pymrst.modules.ad_core.ad_oo.calculate_phase_rate import calculate_phase_rate
from pymrst.modules.ad_core.statefunctions.state_function import StateFunction


class WellPhaseFlux(StateFunction):
    """`q_alpha` per perforation."""

    def __init__(self, model: Any):
        super().__init__(model, label="q_\\alpha")
        self.AutoDiffBackend = getattr(model, "AutoDiffBackend", None)
        self.allowCrossFlow = True
        self.depends_on(["FacilityWellMapping", "PressureGradient", "WellIndex",
                         "Mobility"])

    def evaluate_on_domain(self, model: Any, state: Any) -> List[Any]:
        mapping = self.get_evaluated_dependencies(state, "FacilityWellMapping")
        dp, wi = self.get_evaluated_dependencies(state, "PressureGradient",
                                                 "WellIndex")
        mobw = self.get_evaluated_dependencies(state, "Mobility")
        Tdp = -wi*dp
        return calculate_phase_rate(Tdp, mobw, mapping, self.allowCrossFlow, model)

    evaluateOnDomain = evaluate_on_domain


__all__ = ["WellPhaseFlux"]


if __name__ == "__main__":
    from types import SimpleNamespace

    import numpy as np

    def test_production_is_mobility_times_well_index_times_drawdown():
        mapping = SimpleNamespace(perf2well=np.array([0]), isInjector=np.array([False]),
                                  W=[SimpleNamespace(compi=np.array([1.0, 0.0]))])
        values = {"FacilityWellMapping": mapping, "PressureGradient": np.array([5.0]),
                  "WellIndex": np.array([2.0]),
                  "Mobility": [np.array([1.0]), np.array([3.0])]}
        prop = WellPhaseFlux(SimpleNamespace())
        prop.get_evaluated_dependencies = lambda st, *n: (
            values[n[0]] if len(n) == 1 else [values[k] for k in n])
        q = prop.evaluate_on_domain(SimpleNamespace(verbose=0), {})
        np.testing.assert_allclose(q[0], [-10.0])
        np.testing.assert_allclose(q[1], [-30.0])
        print("ok test_production_is_mobility_times_well_index_times_drawdown")

    test_production_is_mobility_times_well_index_times_drawdown()
    print("\nAll tests passed!")
