"""MRST ``ad-core/statefunctions/flux/ComponentTotalFlux.m``.

Total mass flux of each component, summed over the phases it moves in.
"""

from __future__ import annotations

from typing import Any, List

from pymrst.modules.ad_core.statefunctions.state_function import StateFunction


class ComponentTotalFlux(StateFunction):
    """`V_i = sum_alpha V_{i,alpha}`."""

    def __init__(self, model: Any):
        super().__init__(model, label="V_i")
        self.AutoDiffBackend = model.AutoDiffBackend
        self.depends_on("ComponentPhaseFlux")

    def evaluate_on_domain(self, model: Any, state: Any) -> List[Any]:
        phase_flux = self.get_evaluated_dependencies(state, "ComponentPhaseFlux")
        totals: List[Any] = []
        for row in phase_flux:
            total = None
            for m in row:
                if m is not None:
                    total = m if total is None else total + m
            totals.append(total)
        return totals

    evaluateOnDomain = evaluate_on_domain


__all__ = ["ComponentTotalFlux"]


if __name__ == "__main__":
    from pymrst.modules.ad_core.ad_oo.testing import toy_model, toy_state

    def test_a_model_without_components_has_no_total_flux():
        model = toy_model("WO", nx=3)
        assert model.get_prop(toy_state(model), "ComponentTotalFlux") == []
        print("ok test_a_model_without_components_has_no_total_flux")

    test_a_model_without_components_has_no_total_flux()
    print("\nAll tests passed!")
