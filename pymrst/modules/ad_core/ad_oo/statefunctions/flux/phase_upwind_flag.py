"""MRST ``ad-core/statefunctions/flux/PhaseUpwindFlag.m``.

Which cell is upstream on each face, per phase: `Theta <= 0` means the first
cell of the face.
"""

from __future__ import annotations

from typing import Any, List

import numpy as np

from pymrst.modules.ad_core.statefunctions.state_function import StateFunction
from pymrst.utils.adi import value


class PhaseUpwindFlag(StateFunction):
    """`Theta_alpha <= 0` per face."""

    def __init__(self, model: Any):
        super().__init__(model, label="\\Theta_\\alpha \\leq 0")
        self.AutoDiffBackend = model.AutoDiffBackend
        self.depends_on("PhasePotentialDifference")

    def evaluate_on_domain(self, model: Any, state: Any) -> List[np.ndarray]:
        pot = self.get_evaluated_dependencies(state, "PhasePotentialDifference")
        return [np.asarray(value(p)) <= 0 for p in pot]

    evaluateOnDomain = evaluate_on_domain


__all__ = ["PhaseUpwindFlag"]


if __name__ == "__main__":
    from pymrst.modules.ad_core.ad_oo.testing import toy_model, toy_state

    def test_flow_from_high_to_low_pressure_flags_the_first_cell():
        model = toy_model("W", nx=3)
        state = toy_state(model, pressure=np.array([3.0, 2.0, 5.0]))
        (flag,) = model.get_prop(state, "PhaseUpwindFlag")
        np.testing.assert_array_equal(flag, [True, False])
        print("ok test_flow_from_high_to_low_pressure_flags_the_first_cell")

    test_flow_from_high_to_low_pressure_flags_the_first_cell()
    print("\nAll tests passed!")
