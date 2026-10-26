"""MRST ``ad-core/statefunctions/facility/PerforationComponentPhaseDensity.m``.

Component densities in each phase at the perforated cells, taken from the
reservoir's `ComponentPhaseDensity`.
"""

from __future__ import annotations

from typing import Any, List

from pymrst.modules.ad_core.statefunctions.state_function import StateFunction


class PerforationComponentPhaseDensity(StateFunction):
    """`rho_wc`, `ncomp x nphase`, at each perforated cell."""

    def __init__(self, model: Any):
        super().__init__(model, label="\\rho_{wc}")
        self.AutoDiffBackend = getattr(model, "AutoDiffBackend", None)
        self.depends_on(["ComponentPhaseDensity"], "FlowPropertyFunctions")
        self.depends_on(["FacilityWellMapping"])

    def evaluate_on_domain(self, model: Any, state: Any) -> List[List[Any]]:
        rm = model.ReservoirModel
        rhoc = rm.FlowPropertyFunctions.get(rm, state, "ComponentPhaseDensity")
        mapping = self.get_evaluated_dependencies(state, "FacilityWellMapping")
        return [[None if r is None else r[mapping.cells] for r in row]
                for row in rhoc]

    evaluateOnDomain = evaluate_on_domain


__all__ = ["PerforationComponentPhaseDensity"]


if __name__ == "__main__":
    from types import SimpleNamespace

    import numpy as np

    def test_densities_are_read_at_the_perforated_cells():
        rm = SimpleNamespace(FlowPropertyFunctions=SimpleNamespace(
            get=lambda m, s, n: [[np.array([1.0, 2.0]), None]]))
        prop = PerforationComponentPhaseDensity(SimpleNamespace())
        prop.get_evaluated_dependencies = lambda st, n: SimpleNamespace(
            cells=np.array([1]))
        out = prop.evaluate_on_domain(SimpleNamespace(ReservoirModel=rm), {})
        np.testing.assert_allclose(out[0][0], [2.0])
        assert out[0][1] is None
        print("ok test_densities_are_read_at_the_perforated_cells")

    test_densities_are_read_at_the_perforated_cells()
    print("\nAll tests passed!")
