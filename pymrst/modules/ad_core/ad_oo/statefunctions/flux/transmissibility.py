"""MRST ``ad-core/statefunctions/flux/Transmissibility.m``.

Face transmissibility, times the fluid's pressure-dependent `transMult` of
the face-averaged pressure when the fluid has one.
"""

from __future__ import annotations

from typing import Any
import warnings

import numpy as np

from pymrst.modules.ad_core.statefunctions.state_function import (
    STATE,
    StateFunction,
)
from pymrst.utils.adi import value


class Transmissibility(StateFunction):
    """`T_f`, checked finite at construction."""

    def __init__(self, model: Any):
        super().__init__(model, label="T_f")
        self.AutoDiffBackend = model.AutoDiffBackend
        if getattr(model.fluid, "transMult", None) is not None:
            self.depends_on("pressure", STATE)
        T = getattr(model.operators, "T", None)
        assert T is not None
        T = np.asarray(value(T))
        assert np.all(np.isfinite(T))
        if np.any(T < 0):
            warnings.warn(f"Negative transmissibility in {int(np.sum(T < 0))} interfaces")
        self.output_range = (0.0, np.inf)

    def evaluate_on_domain(self, model: Any, state: Any) -> Any:
        T = model.operators.T
        if getattr(model.fluid, "transMult", None) is not None:
            p = model.get_props(state, "pressure")
            T = model.fluid.transMult(model.operators.avg(p))*T
        return T

    evaluateOnDomain = evaluate_on_domain


__all__ = ["Transmissibility"]


if __name__ == "__main__":
    from pymrst.modules.ad_core.ad_oo.testing import toy_model, toy_state

    def test_transmissibility_is_the_operators_by_default():
        model = toy_model("W", nx=3)
        T = model.get_prop(toy_state(model), "Transmissibility")
        np.testing.assert_allclose(T, model.operators.T)
        print("ok test_transmissibility_is_the_operators_by_default")

    def test_pressure_multiplier_uses_the_face_average():
        model = toy_model("W", nx=2)
        model.fluid.transMult = lambda p: p/1e5
        model.remove_state_function_groupings()
        model.setup_state_function_groupings()
        state = toy_state(model, pressure=np.array([1e5, 3e5]))
        T = model.get_prop(state, "Transmissibility")
        np.testing.assert_allclose(T, 2.0*model.operators.T)
        print("ok test_pressure_multiplier_uses_the_face_average")

    test_transmissibility_is_the_operators_by_default()
    test_pressure_multiplier_uses_the_face_average()
    print("\nAll tests passed!")
