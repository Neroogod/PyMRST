"""MRST ``ad-core/statefunctions/flux/GravityPotentialDifference.m``.

The gravity term `rho_f g dz` on each face, per phase, with the face density
the arithmetic average of the two cells (or a saturation-weighted average).
"""

from __future__ import annotations

from typing import Any, List

import numpy as np

from pymrst.modules.ad_core.statefunctions.state_function import (
    STATE,
    StateFunction,
)

SATURATION_WEIGHT_FLOOR: float = 1e-8


class GravityPotentialDifference(StateFunction):
    """`avg(rho_alpha) * (-g . dz)`; zeros without gravity."""

    saturationWeighting = False

    def __init__(self, model: Any):
        super().__init__(model, label="g \\rho_\\alpha \\Delta z")
        self.AutoDiffBackend = model.AutoDiffBackend
        self.weight = None
        self.depends_on("Density", "PVTPropertyFunctions")
        if self.saturationWeighting:
            self.depends_on("s", STATE)

    def evaluate_on_domain(self, model: Any, state: Any) -> List[Any]:
        nph = int(model.get_active_phases().sum())
        nf = np.shape(model.operators.N)[0]
        if np.linalg.norm(model.gravity) > 0 and nf > 0:
            gdz = -model.get_gravity_gradient()
            names = model.get_phase_names()
            rho = self.get_evaluated_externals(model, state, "Density")
            avg = model.operators.avg
            g_rho_dz = []
            for i in range(nph):
                if self.saturationWeighting:
                    s = model.get_prop(state, f"s{names[i]}")
                    rhof = avg(s*rho[i])/np.maximum(avg(s), SATURATION_WEIGHT_FLOOR)
                else:
                    rhof = avg(rho[i])
                g_rho_dz.append(rhof*gdz)
        else:
            g_rho_dz = [np.zeros(nf) for _ in range(nph)]
        if self.weight is not None:
            g_rho_dz = [self.weight @ x for x in g_rho_dz]
        return g_rho_dz

    evaluateOnDomain = evaluate_on_domain


__all__ = ["GravityPotentialDifference"]


if __name__ == "__main__":
    from pymrst.gridprocessing import cart_grid, compute_geometry
    from pymrst.modules.ad_core.ad_oo.reservoir_model import ReservoirModel
    from pymrst.modules.ad_core.ad_oo.testing import toy_state
    from pymrst.modules.ad_props import init_simple_adi_fluid
    from pymrst.params.gravity.gravity import gravity, isolated_gravity
    from pymrst.params.rock import make_rock

    def _column_model():
        grid = compute_geometry(cart_grid([1, 1, 2], [1.0, 1.0, 2.0]))
        fluid = init_simple_adi_fluid(phases="W", mu=[1e-3], rho=[1000.0],
                                      n=[1.0], c=[0.0])
        model = ReservoirModel(grid, make_rock(grid, 1e-13, 0.3), fluid)
        model.water = True
        return model.validate_model()

    def test_no_gravity_gives_zeros():
        with isolated_gravity("off"):
            model = _column_model()
            (g,) = model.get_prop(toy_state(model), "GravityPotentialDifference")
        np.testing.assert_allclose(g, 0.0)
        print("ok test_no_gravity_gives_zeros")

    def test_gravity_term_is_density_times_g_dz():
        with isolated_gravity("reset", "on"):
            model = _column_model()
            state = toy_state(model, pressure=np.zeros(2))
            (g,) = model.get_prop(state, "GravityPotentialDifference")
            g_norm = np.linalg.norm(gravity())
        # Cell 1 is 1 m below cell 0, so -g.dz across the face is -g.
        np.testing.assert_allclose(g, -1000.0*g_norm)
        print("ok test_gravity_term_is_density_times_g_dz")

    test_no_gravity_gives_zeros()
    test_gravity_term_is_density_times_g_dz()
    print("\nAll tests passed!")
