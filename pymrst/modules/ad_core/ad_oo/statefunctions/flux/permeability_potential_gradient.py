"""MRST ``ad-core/statefunctions/flux/PermeabilityPotentialGradient.m``.

`K grad(Theta_alpha)` on each face, per phase, through a pluggable
permeability-gradient discretisation (TPFA by default).
"""

from __future__ import annotations

from typing import Any, List

from pymrst.modules.ad_core.ad_oo.statefunctions.flux.two_point_flux_approximation import (
    TwoPointFluxApproximation,
)
from pymrst.modules.ad_core.statefunctions.state_function import StateFunction


class PermeabilityPotentialGradient(StateFunction):
    """`T_f Theta_alpha` for TPFA."""

    def __init__(self, model: Any, kgrad: Any):
        super().__init__(model, label="T_f \\Theta_\\alpha")
        self.AutoDiffBackend = model.AutoDiffBackend
        self.PermeabilityGradientDiscretization = kgrad
        self.depends_on("PhasePotentialDifference")
        if isinstance(kgrad, TwoPointFluxApproximation):
            self.depends_on("Transmissibility")

    def evaluate_on_domain(self, model: Any, state: Any) -> List[Any]:
        pot = self.get_evaluated_dependencies(state, "PhasePotentialDifference")
        kgrad = self.PermeabilityGradientDiscretization
        return [kgrad.get_permeability_gradient(model, state, p) for p in pot]

    evaluateOnDomain = evaluate_on_domain


__all__ = ["PermeabilityPotentialGradient"]


if __name__ == "__main__":
    import numpy as np

    from pymrst.modules.ad_core.ad_oo.testing import toy_model, toy_state

    def test_each_phase_potential_is_scaled_by_transmissibility():
        model = toy_model("WO", nx=3)
        state = toy_state(model, pressure=np.array([3e5, 2e5, 1e5]))
        v = model.get_prop(state, "PermeabilityPotentialGradient")
        dp = model.get_prop(state, "PhasePotentialDifference")
        for v_i, dp_i in zip(v, dp):
            np.testing.assert_allclose(v_i, model.operators.T*dp_i)
        print("ok test_each_phase_potential_is_scaled_by_transmissibility")

    test_each_phase_potential_is_scaled_by_transmissibility()
    print("\nAll tests passed!")
