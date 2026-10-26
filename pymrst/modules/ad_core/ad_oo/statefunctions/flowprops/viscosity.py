"""MRST ``ad-core/statefunctions/flowprops/Viscosity.m``.

Phase viscosities, each at its own phase pressure, from the fluid's `muW`,
`muO`, `muG`.
"""

from __future__ import annotations

from typing import Any, List

import numpy as np

from pymrst.modules.ad_core.statefunctions.state_function import StateFunction
from pymrst.modules.ad_core.utils.get_sample_ad import get_sample_ad


class Viscosity(StateFunction):
    """`mu_alpha(p_alpha)` for every active phase."""

    def __init__(self, model: Any, regions: Any = None):
        super().__init__(model, label="\\mu_\\alpha", regions=regions)
        self.AutoDiffBackend = model.AutoDiffBackend
        self.depends_on("PhasePressures")
        self.output_range = (0.0, np.inf)

    def evaluate_on_domain(self, model: Any, state: Any) -> List[Any]:
        names = model.get_phase_names()
        p_phase = self.get_evaluated_dependencies(state, "PhasePressures")
        if not isinstance(p_phase, list):
            p_phase = [p_phase]
        sample, is_ad = get_sample_ad(*p_phase)
        mu = [self.evaluate_phase_viscosity(model, state, names[ph], p_phase[ph])
              for ph in range(len(names))]
        if is_ad:
            mu = [x if hasattr(x, "jac") else
                  model.AutoDiffBackend.convert_to_ad(x, sample) for x in mu]
        return mu

    def evaluate_phase_viscosity(self, model: Any, state: Any, name: str,
                                 p: Any) -> Any:
        return self.evaluate_fluid(model, f"mu{name}", p)

    evaluateOnDomain = evaluate_on_domain
    evaluatePhaseViscosity = evaluate_phase_viscosity


__all__ = ["Viscosity"]


if __name__ == "__main__":
    from pymrst.modules.ad_core.ad_oo.testing import toy_model, toy_state
    from pymrst.utils.adi import init_variables_adi

    def test_constant_viscosities_are_returned_per_phase():
        model = toy_model("WO", nx=3)
        mu_w, mu_o = model.get_prop(toy_state(model), "Viscosity")
        np.testing.assert_allclose(mu_w, 1e-3)
        np.testing.assert_allclose(mu_o, 5e-3)
        print("ok test_constant_viscosities_are_returned_per_phase")

    def test_constant_viscosity_is_lifted_to_ad_with_pressure():
        """A constant viscosity has no derivatives of its own; MRST lifts it so
        it combines with AD relative permeabilities without losing theirs."""
        model = toy_model("WO", nx=3)
        state = toy_state(model)
        (state["pressure"],) = init_variables_adi(state["pressure"])
        mu = model.get_prop(state, "Viscosity")
        assert all(hasattr(x, "jac") for x in mu)
        print("ok test_constant_viscosity_is_lifted_to_ad_with_pressure")

    test_constant_viscosities_are_returned_per_phase()
    test_constant_viscosity_is_lifted_to_ad_with_pressure()
    print("\nAll tests passed!")
