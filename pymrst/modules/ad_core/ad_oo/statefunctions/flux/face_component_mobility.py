"""MRST ``ad-core/statefunctions/flux/FaceComponentMobility.m``.

Component mobility in each phase on each face, taken from the upstream cell.
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


class FaceComponentMobility(StateFunction, UpwindProperty):
    """`lambda_{i,alpha}^f`, `ncomp x nphase`, upwinded per phase."""

    def __init__(self, model: Any, upstr: Optional[Any] = None,
                 upwind_name: str = "PhaseUpwindFlag"):
        if upstr is None:
            upstr = UpwindFunctionWrapperDiscretization(model)
        StateFunction.__init__(self, model, label="\\lambda_{i, \\alpha}^f")
        self.AutoDiffBackend = model.AutoDiffBackend
        self.init_upwind_property(upstr)
        self.upwind_name = upwind_name
        self.depends_on(upwind_name)
        self.depends_on("ComponentMobility", "FlowPropertyFunctions")

    def evaluate_on_domain(self, model: Any, state: Any) -> List[List[Any]]:
        flag = self.get_evaluated_dependencies(state, self.upwind_name)
        mob = self.get_evaluated_externals(model, state, "ComponentMobility")
        return [[None if m is None else
                 self.face_upstream(model, state, flag[ph], m)
                 for ph, m in enumerate(row)] for row in mob]

    evaluateOnDomain = evaluate_on_domain


__all__ = ["FaceComponentMobility"]


if __name__ == "__main__":
    import numpy as np

    from pymrst.modules.ad_core.ad_oo.testing import toy_model, toy_state

    def test_a_model_without_components_has_none():
        model = toy_model("WO", nx=3)
        state = toy_state(model)
        assert model.get_prop(state, "FaceComponentMobility") == []
        print("ok test_a_model_without_components_has_none")

    test_a_model_without_components_has_none()
    print("\nAll tests passed!")
