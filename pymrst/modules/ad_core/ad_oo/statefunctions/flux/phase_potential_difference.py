"""MRST ``ad-core/statefunctions/flux/PhasePotentialDifference.m``.

`Theta_alpha = grad(p_alpha) + rho_alpha g dz` per face and phase - the
potential that drives each phase across a face.
"""

from __future__ import annotations

from typing import Any, List

import numpy as np

from pymrst.modules.ad_core.statefunctions.state_function import StateFunction
from pymrst.utils.adi import value


class PhasePotentialDifference(StateFunction):
    """Pressure gradient plus the gravity term, when there is gravity."""

    def __init__(self, model: Any):
        super().__init__(model, label="\\Theta_\\alpha")
        self.AutoDiffBackend = model.AutoDiffBackend
        self.depends_on("PressureGradient")
        self.hasGravity = bool(np.linalg.norm(
            np.ravel(value(model.get_gravity_gradient())), np.inf) > 0)
        if self.hasGravity:
            self.depends_on("GravityPotentialDifference")

    def evaluate_on_domain(self, model: Any, state: Any) -> List[Any]:
        dp = self.get_evaluated_dependencies(state, "PressureGradient")
        v = list(dp)
        if self.hasGravity:
            rho_g_dz = self.get_evaluated_dependencies(
                state, "GravityPotentialDifference")
            v = [v[i] + rho_g_dz[i] for i in range(len(dp))]
        return v

    evaluateOnDomain = evaluate_on_domain


__all__ = ["PhasePotentialDifference"]


if __name__ == "__main__":
    from pymrst.modules.ad_core.ad_oo.testing import toy_model, toy_state

    def test_horizontal_potential_is_the_pressure_gradient():
        model = toy_model("WO", nx=3)
        state = toy_state(model, pressure=np.array([3.0, 2.0, 0.0]))
        pot = model.get_prop(state, "PhasePotentialDifference")
        for p in pot:
            np.testing.assert_allclose(p, [-1.0, -2.0])
        print("ok test_horizontal_potential_is_the_pressure_gradient")

    test_horizontal_potential_is_the_pressure_gradient()
    print("\nAll tests passed!")
