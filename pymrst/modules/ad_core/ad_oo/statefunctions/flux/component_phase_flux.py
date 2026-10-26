"""MRST ``ad-core/statefunctions/flux/ComponentPhaseFlux.m``.

Mass flux of each component in each phase: `-lambda_{i,alpha}^f T Theta_alpha`.
"""

from __future__ import annotations

from typing import Any, List

from pymrst.modules.ad_core.statefunctions.state_function import StateFunction


class ComponentPhaseFlux(StateFunction):
    """`V_{i,alpha}`, `ncomp x nphase`."""

    def __init__(self, model: Any, mob_name: str = "FaceComponentMobility"):
        super().__init__(model, label="V_{i,\\alpha}")
        self.AutoDiffBackend = model.AutoDiffBackend
        self.mobility_name = mob_name
        self.depends_on(["PermeabilityPotentialGradient", self.mobility_name])

    def evaluate_on_domain(self, model: Any, state: Any) -> List[List[Any]]:
        kgrad, comp_mob = self.get_evaluated_dependencies(
            state, "PermeabilityPotentialGradient", self.mobility_name)
        return [[None if mob is None else -mob*kgrad[ph]
                 for ph, mob in enumerate(row)] for row in comp_mob]

    evaluateOnDomain = evaluate_on_domain


__all__ = ["ComponentPhaseFlux"]


if __name__ == "__main__":
    from pymrst.modules.ad_core.ad_oo.testing import toy_model, toy_state

    def test_a_model_without_components_has_no_component_flux():
        model = toy_model("WO", nx=3)
        assert model.get_prop(toy_state(model), "ComponentPhaseFlux") == []
        print("ok test_a_model_without_components_has_no_component_flux")

    test_a_model_without_components_has_no_component_flux()
    print("\nAll tests passed!")
