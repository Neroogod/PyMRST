"""MRST ``ad-core/statefunctions/flowprops/RsMax.m``.

The bubble-point limit: the most gas oil can dissolve at the cell pressure,
the fluid's `rsSat(p)`.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from pymrst.modules.ad_core.statefunctions.state_function import (
    STATE,
    StateFunction,
)


class RsMax(StateFunction):
    """`R_s^max = rsSat(p)`."""

    def __init__(self, model: Any, regions: Any = None):
        super().__init__(model, label="R_s^{max}", regions=regions)
        self.AutoDiffBackend = model.AutoDiffBackend
        self.depends_on("pressure", STATE)
        self.output_range = (0.0, np.inf)

    def evaluate_on_domain(self, model: Any, state: Any) -> Any:
        p = model.get_prop(state, "pressure")
        return self.evaluate_fluid(model, "rsSat", p)

    evaluateOnDomain = evaluate_on_domain


__all__ = ["RsMax"]


if __name__ == "__main__":
    from pymrst.modules.ad_core.ad_oo.testing import toy_model, toy_state

    def test_rs_max_reads_the_saturated_curve_at_cell_pressure():
        model = toy_model("WOG", nx=2)
        model.fluid.rsSat = lambda p: 1e-5*p
        state = toy_state(model, pressure=np.array([1e5, 2e5]))
        rs = RsMax(model).evaluate_on_domain(model, state)
        np.testing.assert_allclose(rs, [1.0, 2.0])
        print("ok test_rs_max_reads_the_saturated_curve_at_cell_pressure")

    def test_rs_max_dispatches_per_pvt_region():
        model = toy_model("WOG", nx=2)
        model.fluid.rsSat = [lambda p: 1e-5*p, lambda p: 2e-5*p]
        state = toy_state(model, pressure=np.array([1e5, 1e5]))
        rs = RsMax(model, np.array([1, 2])).evaluate_on_domain(model, state)
        np.testing.assert_allclose(rs, [1.0, 2.0])
        print("ok test_rs_max_dispatches_per_pvt_region")

    test_rs_max_reads_the_saturated_curve_at_cell_pressure()
    test_rs_max_dispatches_per_pvt_region()
    print("\nAll tests passed!")
