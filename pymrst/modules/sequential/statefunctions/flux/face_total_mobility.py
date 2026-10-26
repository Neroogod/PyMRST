"""MRST ``sequential/statefunctions/flux/FaceTotalMobility.m``.

The total mobility on each interior face: the sum of the phases' face
mobilities (`FaceMobility`, or another named function).
"""

from __future__ import annotations

from typing import Any, List

import numpy as np

from pymrst.modules.ad_core.statefunctions.state_function import StateFunction


class FaceTotalMobility(StateFunction):
    """MRST `FaceTotalMobility`."""

    def __init__(self, model: Any, mobility_name: str = "FaceMobility"):
        super().__init__(model)
        self.depends_on(mobility_name)
        self.mobility_name = mobility_name

    def evaluate_on_domain(self, model: Any, state: Any) -> Any:
        mob = self.get_evaluated_dependencies(state, self.mobility_name)
        mobT = np.zeros(np.size(model.operators.T))
        for m in mob:
            mobT = mobT + m
        return mobT

    evaluateOnDomain = evaluate_on_domain


__all__ = ["FaceTotalMobility"]


if __name__ == "__main__":
    import warnings

    import numpy as np

    from pymrst.modules.sequential.testing import toy_case, toy_transport_state
    from pymrst.utils.adi import value

    warnings.simplefilter("ignore")
    T = toy_transport_state(toy_case())

    def test_it_sums_the_face_mobilities():
        mob = T.parent.get_prop(T.ad_state, "FaceMobility")
        total = T.parent.get_prop(T.ad_state, "FaceTotalMobility")
        np.testing.assert_allclose(value(total), sum(value(m) for m in mob))
        print("ok test_it_sums_the_face_mobilities")

    test_it_sums_the_face_mobilities()
    print("\nAll tests passed!")
