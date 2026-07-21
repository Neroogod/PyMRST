"""Phase density (MRST ad-core/statefunctions/flowprops).

Ports MRST's `BlackOilDensity.m` and `DensityDerivedShrinkageFactors.m`.

`BlackOilDensity` builds the reservoir phase densities from the shrinkage
factors and surface densities, ``rho_alpha = b_alpha * rhoS_alpha``, then adds
the mass the oil carries as dissolved gas (``rs``) and the gas carries as
vaporised oil (``rv``):

    rho_O = b_O (rhoS_O + rs * rhoS_G)
    rho_G = b_G (rhoS_G + rv * rhoS_O)

`DensityDerivedShrinkageFactors` is the inverse relationship for models that
provide density directly: ``b_alpha = rho_alpha / rhoS_alpha``.

Both reproduce the pre-split PyMRST `Density` statefunction (see
`validate_phase340.py`), which folded the carry terms into a single class.
"""

from __future__ import annotations

from typing import Any, List

from pymrst.modules.ad_core.statefunctions.state_function import (
    STATE,
    StateFunction,
)


class BlackOilDensity(StateFunction):
    """Reservoir phase densities, with dissolved-gas / vaporised-oil carry."""

    def __init__(self, model: Any = None, label: str = "\\rho_\\alpha"):
        super().__init__(model, label)
        self.disgas = model is not None and model.fluid.disgas
        self.vapoil = model is not None and model.fluid.vapoil
        self.depends_on(["ShrinkageFactors", "SurfaceDensity"])
        if self.disgas:
            self.depends_on("rs", STATE)
        if self.vapoil:
            self.depends_on("rv", STATE)

    def evaluate_on_domain(self, model: Any, state: Any) -> List[Any]:
        b = self.get_evaluated(state, "ShrinkageFactors")
        rho_s = self.get_evaluated(state, "SurfaceDensity")
        nph = model.n_phases
        rho = [b[i] * rho_s[i] for i in range(nph)]
        # oil is phase 1, gas is phase 2 in the water/oil/gas ordering
        if self.disgas and nph == 3:
            rho[1] = rho[1] + state["rs"] * b[1] * rho_s[2]
        if self.vapoil and nph == 3:
            rho[2] = rho[2] + state["rv"] * b[2] * rho_s[1]
        return rho


class DensityDerivedShrinkageFactors(StateFunction):
    """``b_alpha = rho_alpha / rhoS_alpha`` for density-first models."""

    def __init__(self, model: Any = None, label: str = "b_\\alpha"):
        super().__init__(model, label)
        self.depends_on(["Density", "SurfaceDensity"])

    def evaluate_on_domain(self, model: Any, state: Any) -> List[Any]:
        rho = self.get_evaluated(state, "Density")
        rho_s = self.get_evaluated(state, "SurfaceDensity")
        return [rho[i] / rho_s[i] for i in range(len(rho))]


__all__ = ["BlackOilDensity", "DensityDerivedShrinkageFactors"]


if __name__ == "__main__":
    import numpy as np

    from pymrst.gridprocessing import cart_grid, compute_geometry
    from pymrst.modules.ad_blackoil.models.generic_black_oil_model import (
        GenericBlackOilModel,
    )
    from pymrst.modules.ad_core.statefunctions.flowprops.shrinkage_factors import (
        BlackOilShrinkageFactors,
    )
    from pymrst.modules.ad_core.statefunctions.flowprops.surface_density import (
        SurfaceDensity,
    )
    from pymrst.modules.ad_core.statefunctions.state_function import State
    from pymrst.modules.ad_props import init_simple_adi_fluid
    from pymrst.params import add_well, make_rock
    from pymrst.utils.adi import value
    from pymrst.utils.units import BARSA, CENTI, DARCY, MILLI, POISE

    def _ow_model():
        g = compute_geometry(cart_grid([5, 4, 2], [250.0, 200.0, 20.0]))
        rock = make_rock(g, 100 * MILLI * DARCY, 0.25)
        fluid = init_simple_adi_fluid(
            mu=np.array([1.0, 5.0]) * CENTI * POISE, rho=[1000.0, 800.0],
            n=[2, 2], c=[1e-6 / BARSA, 1e-4 / BARSA], p_ref=200 * BARSA)
        w = add_well(None, g, rock, np.array([0]), well_type="rate", val=2e-3,
                     compi=np.array([1.0, 0.0]), sign=1)
        w = add_well(w, g, rock, np.array([g.cells.num - 1]), well_type="bhp",
                     val=150 * BARSA, compi=np.array([0.0, 1.0]), sign=-1)
        model = GenericBlackOilModel(g, rock, fluid, w)
        # register the flowprops the density depends on
        model.PVTPropertyFunctions.set_state_function(
            "ShrinkageFactors", BlackOilShrinkageFactors(model))
        model.PVTPropertyFunctions.set_state_function(
            "SurfaceDensity", SurfaceDensity(model))
        st = model.init_state_ad(State(
            pressure=np.linspace(210, 190, g.cells.num) * BARSA,
            sW=np.full(g.cells.num, 0.45), qWs=np.array([2e-3]),
            qOs=np.array([-2e-3]), bhp=np.array([150 * BARSA])))
        return model, st

    def test_density_matches_legacy_on_oil_water():
        """rho = b * rhoS reproduces the pre-split Density (no carry here)."""
        from pymrst.modules.ad_core.statefunctions.pvt_property_functions import (
            Density as LegacyDensity,
        )
        model, st = _ow_model()
        model.PVTPropertyFunctions.set_state_function(
            "BlackOilDensity", BlackOilDensity(model))
        new = model.get_props(st, "BlackOilDensity")
        model.PVTPropertyFunctions.set_state_function(
            "Density", LegacyDensity(model))
        old = model.get_props(st, "Density")
        for ph in range(model.n_phases):
            assert np.allclose(value(new[ph]), value(old[ph]))
            assert np.allclose((new[ph] - old[ph]).jac[0].toarray(), 0.0)
        print("ok test_density_matches_legacy_on_oil_water")

    def test_density_derived_inverts_density():
        """DensityDerivedShrinkageFactors recovers b from rho: rho/rhoS."""
        model, st = _ow_model()
        model.PVTPropertyFunctions.set_state_function(
            "Density", BlackOilDensity(model))
        model.PVTPropertyFunctions.set_state_function(
            "DDShrink", DensityDerivedShrinkageFactors(model))
        b_direct = model.get_props(st, "ShrinkageFactors")
        b_derived = model.get_props(st, "DDShrink")
        for ph in range(model.n_phases):
            assert np.allclose(value(b_derived[ph]), value(b_direct[ph]))
        print("ok test_density_derived_inverts_density")

    test_density_matches_legacy_on_oil_water()
    test_density_derived_inverts_density()
    print("\nAll tests passed!")
