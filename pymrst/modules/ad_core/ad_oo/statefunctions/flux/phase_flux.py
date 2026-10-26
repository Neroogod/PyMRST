"""MRST ``ad-core/statefunctions/flux/PhaseFlux.m``.

Volumetric (reservoir) flux of each phase across each face:
`v_alpha = -lambda_alpha^f T Theta_alpha`.
"""

from __future__ import annotations

from typing import Any, List

from pymrst.modules.ad_core.statefunctions.state_function import StateFunction


class PhaseFlux(StateFunction):
    """`v_alpha` per face."""

    def __init__(self, model: Any):
        super().__init__(model, label="v_\\alpha")
        self.AutoDiffBackend = model.AutoDiffBackend
        self.depends_on(["FaceMobility", "PermeabilityPotentialGradient"])

    def evaluate_on_domain(self, model: Any, state: Any) -> List[Any]:
        mob, kgrad = self.get_evaluated_dependencies(
            state, "FaceMobility", "PermeabilityPotentialGradient")
        return [-mob[i]*kgrad[i] for i in range(len(mob))]

    evaluateOnDomain = evaluate_on_domain


__all__ = ["PhaseFlux"]


if __name__ == "__main__":
    import numpy as np

    from pymrst.modules.ad_core.ad_oo.testing import toy_model, toy_state

    def test_flux_runs_down_the_pressure_gradient():
        model = toy_model("W", nx=3)
        state = toy_state(model, pressure=np.array([3e5, 2e5, 1e5]))
        (v,) = model.get_prop(state, "PhaseFlux")
        mob = 1.0/1e-3
        np.testing.assert_allclose(v, mob*model.operators.T*1e5)
        print("ok test_flux_runs_down_the_pressure_gradient")

    test_flux_runs_down_the_pressure_gradient()
    print("\nAll tests passed!")
