"""Phase viscosity (MRST ad-core/statefunctions/flowprops).

Ports MRST's `Viscosity.m` (pressure-only base) and `BlackOilViscosity.m`
(live-oil / wet-gas). ``mu_alpha`` enters every mobility, so it sits directly
under `Mobility` in the flow-property graph.

Split on MRST's line: the base evaluates each phase viscosity at its own
pressure; the black-oil subclass additionally passes the dissolved ``rs`` /
vaporised ``rv`` and the saturated-phase flags, so the oil viscosity follows
the live-oil branch and the gas viscosity the wet-gas branch. Together they
reproduce the pre-split `Viscosity` statefunction (see `validate_phase340.py`).
"""

from __future__ import annotations

from typing import Any

from pymrst.modules.ad_core.statefunctions.flowprops.blackoil_pvt_utils import (
    phase_pressures,
    pvt_args,
)
from pymrst.modules.ad_core.statefunctions.state_function import (
    STATE,
    StateFunction,
)


class Viscosity(StateFunction):
    """``mu_alpha`` for a pressure-only fluid."""

    def __init__(self, model: Any = None, label: str = "\\mu_\\alpha"):
        super().__init__(model, label)
        self.use_phase_pressures = model is not None and model.fluid.has_pc
        if self.use_phase_pressures:
            self.depends_on("PhasePressures")
        else:
            self.depends_on("pressure", STATE)

    def evaluate_on_domain(self, model: Any, state: Any):
        p_phase = phase_pressures(self, model, state)
        if not self.use_phase_pressures:
            return list(model.fluid.mu(p_phase[0]))
        return [model.fluid.mu(p_phase[i])[i] for i in range(model.n_phases)]


class BlackOilViscosity(Viscosity):
    """``mu_alpha`` for a black-oil fluid: live-oil and wet-gas branches."""

    def __init__(self, model: Any = None):
        super().__init__(model)
        self.disgas = model is not None and model.fluid.disgas
        self.vapoil = model is not None and model.fluid.vapoil
        if self.disgas:
            self.depends_on(["rs", "status"], STATE)
        if self.vapoil:
            self.depends_on(["rv", "status"], STATE)

    def evaluate_on_domain(self, model: Any, state: Any):
        p_phase = phase_pressures(self, model, state)
        _, rs, sat, rv, sat_g = pvt_args(model, state)
        if not self.use_phase_pressures:
            return list(model.fluid.mu(p_phase[0], rs, sat, rv, sat_g))
        return [model.fluid.mu(p_phase[i], rs, sat, rv, sat_g)[i]
                for i in range(model.n_phases)]


__all__ = ["BlackOilViscosity", "Viscosity"]


if __name__ == "__main__":
    import numpy as np

    from pymrst.gridprocessing import cart_grid, compute_geometry
    from pymrst.modules.ad_blackoil.models.generic_black_oil_model import (
        GenericBlackOilModel,
    )
    from pymrst.modules.ad_core.statefunctions.pvt_property_functions import (
        Viscosity as LegacyViscosity,
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
        st = model.init_state_ad(State(
            pressure=np.linspace(210, 190, g.cells.num) * BARSA,
            sW=np.full(g.cells.num, 0.45), qWs=np.array([2e-3]),
            qOs=np.array([-2e-3]), bhp=np.array([150 * BARSA])))
        return model, st

    def test_base_matches_legacy_on_oil_water():
        model, st = _ow_model()
        new = Viscosity(model).evaluate_on_domain(model, st)
        model.PVTPropertyFunctions.set_state_function(
            "Viscosity", LegacyViscosity(model))
        old = model.get_props(st, "Viscosity")
        for ph in range(model.n_phases):
            assert np.allclose(value(new[ph]), value(old[ph]))
        print("ok test_base_matches_legacy_on_oil_water")

    test_base_matches_legacy_on_oil_water()
    print("\nAll tests passed!")
