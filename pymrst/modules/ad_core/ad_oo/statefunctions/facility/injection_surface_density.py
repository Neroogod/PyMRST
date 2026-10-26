"""MRST ``ad-core/statefunctions/facility/InjectionSurfaceDensity.m``.

Surface density of what each well injects: the reservoir surface density at
the well's top cell, overridden by the well's own `rhoS` for injectors that
specify one.
"""

from __future__ import annotations

from typing import Any, List

import numpy as np

from pymrst.modules.ad_core.statefunctions.state_function import StateFunction


class InjectionSurfaceDensity(StateFunction):
    """`rho_alpha^w` per active well."""

    def __init__(self, model: Any):
        super().__init__(model, label="\\rho_\\alpha^{w}")
        self.AutoDiffBackend = getattr(model, "AutoDiffBackend", None)
        self.depends_on("FacilityWellMapping")
        self.depends_on("SurfaceDensity", "PVTPropertyFunctions")

    def evaluate_on_domain(self, facility: Any, state: Any) -> List[np.ndarray]:
        model = facility.ReservoirModel
        mapping = self.get_evaluated_dependencies(state, "FacilityWellMapping")
        rho_sr = model.PVTPropertyFunctions.get(model, state, "SurfaceDensity")
        W = mapping.W
        topcell = np.array([np.asarray(w.cells).ravel()[0] for w in W], dtype=np.int64)
        rho_s = [np.array(np.asarray(r)[topcell], dtype=float) for r in rho_sr]
        is_inj = np.asarray(mapping.isInjector, dtype=bool)
        if any(getattr(w, "rhoS", None) is not None for w in W) and np.any(is_inj):
            rho_s_inj = np.vstack([np.asarray(W[i].rhoS, dtype=float).ravel()
                                   for i in np.flatnonzero(is_inj)])
            for i in range(len(rho_s)):
                rho_s[i][is_inj] = rho_s_inj[:, i]
        return rho_s

    evaluateOnDomain = evaluate_on_domain


__all__ = ["InjectionSurfaceDensity"]


if __name__ == "__main__":
    from types import SimpleNamespace

    def test_injector_with_its_own_surface_density_overrides_the_reservoir():
        W = [SimpleNamespace(cells=np.array([1, 2]), rhoS=np.array([1050.0, 800.0])),
             SimpleNamespace(cells=np.array([0]), rhoS=None)]
        model = SimpleNamespace(PVTPropertyFunctions=SimpleNamespace(
            get=lambda m, s, n: [np.array([1000.0, 1001.0, 1002.0]),
                                 np.array([700.0, 701.0, 702.0])]))
        prop = InjectionSurfaceDensity(SimpleNamespace())
        prop.get_evaluated_dependencies = lambda st, n: SimpleNamespace(
            W=W, isInjector=np.array([True, False]))
        rho = prop.evaluate_on_domain(SimpleNamespace(ReservoirModel=model), {})
        np.testing.assert_allclose(rho[0], [1050.0, 1000.0])
        np.testing.assert_allclose(rho[1], [800.0, 700.0])
        print("ok test_injector_with_its_own_surface_density_overrides_the_reservoir")

    test_injector_with_its_own_surface_density_overrides_the_reservoir()
    print("\nAll tests passed!")
