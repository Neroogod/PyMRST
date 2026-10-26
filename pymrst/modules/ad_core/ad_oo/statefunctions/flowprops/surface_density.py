"""MRST ``ad-core/statefunctions/flowprops/SurfaceDensity.m``.

Phase densities at surface conditions, one column per active phase, picked
per cell from the model's surface densities by PVT region.
"""

from __future__ import annotations

from typing import Any, List

import numpy as np

from pymrst.modules.ad_core.statefunctions.state_function import StateFunction


class SurfaceDensity(StateFunction):
    """`rho_alpha^S` in every cell."""

    def __init__(self, model: Any, regions: Any = None):
        super().__init__(model, label="\\rho_{\\alpha S}", regions=regions)
        self.AutoDiffBackend = model.AutoDiffBackend
        self.output_range = (0.0, np.inf)

    def evaluate_on_domain(self, model: Any, state: Any) -> List[np.ndarray]:
        rho_s = model.get_surface_densities(self.regions)
        return [rho_s[:, i] for i in range(rho_s.shape[1])]

    evaluateOnDomain = evaluate_on_domain


__all__ = ["SurfaceDensity"]


if __name__ == "__main__":
    from pymrst.modules.ad_core.ad_oo.testing import toy_model, toy_state

    def test_one_column_per_phase_one_row_per_cell():
        model = toy_model("WO", nx=3)
        rho_s = SurfaceDensity(model, np.ones(3, dtype=int)).evaluate_on_domain(
            model, toy_state(model))
        assert len(rho_s) == 2
        np.testing.assert_allclose(rho_s[0], 1000.0)
        np.testing.assert_allclose(rho_s[1], 700.0)
        print("ok test_one_column_per_phase_one_row_per_cell")

    test_one_column_per_phase_one_row_per_cell()
    print("\nAll tests passed!")
