"""MRST ``ad-core/statefunctions/facility/PerforationMobility.m``.

Phase mobilities of the perforated cells, taken from the reservoir's
`Mobility`.
"""

from __future__ import annotations

from typing import Any, List

from pymrst.modules.ad_core.statefunctions.state_function import StateFunction


class PerforationMobility(StateFunction):
    """`lambda_wc`: reservoir mobility at each perforated cell."""

    def __init__(self, model: Any):
        super().__init__(model, label="\\lambda_{wc}")
        self.AutoDiffBackend = getattr(model, "AutoDiffBackend", None)
        self.depends_on(["FacilityWellMapping"])
        self.depends_on(["Mobility"], "FlowPropertyFunctions")

    def evaluate_on_domain(self, model: Any, state: Any) -> List[Any]:
        rm = model.ReservoirModel
        mapping = self.get_evaluated_dependencies(state, "FacilityWellMapping")
        mob = rm.FlowPropertyFunctions.get(rm, state, "Mobility")
        if not isinstance(mob, list):
            mob = [mob]
        return [m[mapping.cells] for m in mob]

    evaluateOnDomain = evaluate_on_domain


__all__ = ["PerforationMobility"]


if __name__ == "__main__":
    from types import SimpleNamespace

    import numpy as np

    def test_mobility_is_read_at_the_perforated_cells():
        rm = SimpleNamespace(FlowPropertyFunctions=SimpleNamespace(
            get=lambda m, s, n: [np.array([1.0, 2.0, 3.0])]))
        prop = PerforationMobility(SimpleNamespace())
        prop.get_evaluated_dependencies = lambda st, n: SimpleNamespace(
            cells=np.array([2, 0]))
        (mob,) = prop.evaluate_on_domain(SimpleNamespace(ReservoirModel=rm), {})
        np.testing.assert_allclose(mob, [3.0, 1.0])
        print("ok test_mobility_is_read_at_the_perforated_cells")

    test_mobility_is_read_at_the_perforated_cells()
    print("\nAll tests passed!")
