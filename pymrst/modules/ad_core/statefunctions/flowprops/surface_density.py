"""Surface density (MRST ad-core/statefunctions/flowprops).

Ports MRST's `SurfaceDensity.m`: the phase densities at surface conditions,
one constant per phase, broadcast to every cell. It is the ``rhoS`` that
`BlackOilDensity` multiplies by the shrinkage factor to get the reservoir
density, and the ``rhoS`` a well uses to convert reservoir to surface rates.

In MRST it is a state function (not a bare constant) so that a compositional or
thermal model can make it vary; for black-oil it is the fixed
``model.fluid.rho_s`` per phase.
"""

from __future__ import annotations

from typing import Any, List

import numpy as np

from pymrst.modules.ad_core.statefunctions.state_function import StateFunction


class SurfaceDensity(StateFunction):
    """Per-phase surface densities, one value per phase, per cell."""

    def __init__(self, model: Any = None, label: str = "\\rho_{\\alpha S}"):
        super().__init__(model, label)

    def evaluate_on_domain(self, model: Any, state: Any) -> List[np.ndarray]:
        rho_s = np.asarray(model.fluid.rho_s, dtype=float)
        nc = model.G.cells.num
        return [np.full(nc, rho_s[i]) for i in range(model.n_phases)]


__all__ = ["SurfaceDensity"]


if __name__ == "__main__":
    from pymrst.gridprocessing import cart_grid, compute_geometry
    from pymrst.modules.ad_blackoil.models.generic_black_oil_model import (
        GenericBlackOilModelAD,
    )
    from pymrst.modules.ad_core.statefunctions.state_function import State
    from pymrst.modules.ad_props import init_simple_adi_fluid
    from pymrst.params import add_well, make_rock
    from pymrst.utils.units import BARSA, CENTI, DARCY, MILLI, POISE

    def _ow_model():
        g = compute_geometry(cart_grid([4, 3, 2], [200.0, 150.0, 20.0]))
        rock = make_rock(g, 100 * MILLI * DARCY, 0.25)
        fluid = init_simple_adi_fluid(
            mu=np.array([1.0, 5.0]) * CENTI * POISE, rho=[1000.0, 800.0],
            n=[2, 2], c=[1e-6 / BARSA, 1e-4 / BARSA], p_ref=200 * BARSA)
        w = add_well(None, g, rock, np.array([0]), well_type="rate", val=2e-3,
                     compi=np.array([1.0, 0.0]), sign=1)
        model = GenericBlackOilModelAD(g, rock, fluid, w)
        return model, g.cells.num

    def test_surface_density_broadcasts_rho_s():
        """Each phase gets its constant surface density in every cell."""
        model, nc = _ow_model()
        rhoS = SurfaceDensity(model).evaluate_on_domain(model, State())
        assert len(rhoS) == model.n_phases
        for i in range(model.n_phases):
            assert rhoS[i].shape == (nc,)
            assert np.allclose(rhoS[i], model.fluid.rho_s[i])
        print("ok test_surface_density_broadcasts_rho_s")

    test_surface_density_broadcasts_rho_s()
    print("\nAll tests passed!")
