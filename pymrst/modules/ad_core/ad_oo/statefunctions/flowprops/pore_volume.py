"""MRST ``ad-core/statefunctions/flowprops/PoreVolume.m``.

The static pore volume, straight from the model's operators.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from pymrst.modules.ad_core.statefunctions.state_function import StateFunction
from pymrst.utils.adi import value


class PoreVolume(StateFunction):
    """`Phi`: the operators' pore volume, checked at construction."""

    def __init__(self, model: Any, regions: Any = None):
        super().__init__(model, label="\\Phi", regions=regions)
        self.AutoDiffBackend = model.AutoDiffBackend
        pv = getattr(model.operators, "pv", None)
        assert pv is not None, (
            "Pore-volume (pv) must be present as field in operators struct")
        assert np.size(value(pv)) == model.G.cells.num, (
            "Pore-volumes must be defined in each cell.")
        assert np.all(np.asarray(value(pv)) > 0), (
            "Pore-volumes must be non-negative.")

    def evaluate_on_domain(self, model: Any, state: Any) -> Any:
        return model.operators.pv

    evaluateOnDomain = evaluate_on_domain


__all__ = ["PoreVolume"]


if __name__ == "__main__":
    from pymrst.modules.ad_core.ad_oo.testing import toy_model, toy_state

    def test_pore_volume_is_porosity_times_volume():
        model = toy_model("W", nx=3)
        pv = PoreVolume(model).evaluate_on_domain(model, toy_state(model))
        np.testing.assert_allclose(pv, 0.3)
        print("ok test_pore_volume_is_porosity_times_volume")

    def test_a_zero_pore_volume_is_refused():
        model = toy_model("W", nx=3)
        model.operators.pv = np.array([0.3, 0.0, 0.3])
        try:
            PoreVolume(model)
            raise SystemExit("should have raised")
        except AssertionError:
            pass
        print("ok test_a_zero_pore_volume_is_refused")

    test_pore_volume_is_porosity_times_volume()
    test_a_zero_pore_volume_is_refused()
    print("\nAll tests passed!")
