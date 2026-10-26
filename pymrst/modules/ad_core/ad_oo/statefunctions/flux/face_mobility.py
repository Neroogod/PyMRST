"""MRST ``ad-core/statefunctions/flux/FaceMobility.m``.

Phase mobility on each face, taken from the upstream cell.
"""

from __future__ import annotations

from typing import Any, List, Optional

from pymrst.modules.ad_core.ad_oo.statefunctions.flux.upwind_function_wrapper_discretization import (
    UpwindFunctionWrapperDiscretization,
)
from pymrst.modules.ad_core.ad_oo.statefunctions.flux.upwind_property import (
    UpwindProperty,
)
from pymrst.modules.ad_core.statefunctions.state_function import StateFunction


class FaceMobility(StateFunction, UpwindProperty):
    """`lambda_alpha^f`, single-point upwinded by `PhaseUpwindFlag`."""

    def __init__(self, model: Any, upstr: Optional[Any] = None,
                 upwind_name: str = "PhaseUpwindFlag"):
        if upstr is None:
            upstr = UpwindFunctionWrapperDiscretization(model)
        StateFunction.__init__(self, model, label="\\lambda_\\alpha^f")
        self.AutoDiffBackend = model.AutoDiffBackend
        self.init_upwind_property(upstr)
        self.upwind_name = upwind_name
        self.depends_on(upwind_name)
        self.depends_on("Mobility", "FlowPropertyFunctions")

    def evaluate_on_domain(self, model: Any, state: Any) -> List[Any]:
        flag = self.get_evaluated_dependencies(state, self.upwind_name)
        mob = self.get_evaluated_externals(model, state, "Mobility")
        return [self.face_upstream(model, state, flag[i], mob[i])
                for i in range(len(mob))]

    evaluateOnDomain = evaluate_on_domain


__all__ = ["FaceMobility"]


if __name__ == "__main__":
    import numpy as np

    from pymrst.modules.ad_core.ad_oo.testing import toy_model, toy_state

    def test_face_mobility_comes_from_the_upstream_cell():
        model = toy_model("WO", nx=3)
        state = toy_state(model, pressure=np.array([3.0, 2.0, 5.0]),
                          s=np.array([[0.1, 0.9], [0.5, 0.5], [0.9, 0.1]]))
        fmob_w, _ = model.get_prop(state, "FaceMobility")
        mob_w, _ = model.get_prop(state, "Mobility")
        np.testing.assert_allclose(fmob_w, [mob_w[0], mob_w[2]])
        print("ok test_face_mobility_comes_from_the_upstream_cell")

    test_face_mobility_comes_from_the_upstream_cell()
    print("\nAll tests passed!")
