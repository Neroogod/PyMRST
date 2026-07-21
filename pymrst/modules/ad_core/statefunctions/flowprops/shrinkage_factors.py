"""Shrinkage factors (MRST ad-core/statefunctions/flowprops).

Ports MRST's `ShrinkageFactors.m` (pressure-only, the immiscible/dead-oil base)
and `BlackOilShrinkageFactors.m` (live-oil / wet-gas, with dissolved `rs` and
vaporised `rv`). The shrinkage factor ``b_alpha`` converts a phase's reservoir
volume to surface volume - the reciprocal of the formation-volume factor - and
is the quantity every black-oil accumulation and flux term is written in.

PyMRST used to compute this inside one monolithic statefunction that folded the
black-oil case into the base. This module splits it on MRST's own line: a
pressure-only base, and a subclass that adds the saturated-state read. The two
together reproduce the pre-split statefunction bit-for-bit (proved in
`validate_phase340.py`), so black-oil PVT now composes through the graph the
way `ad-eor` and `solvent` need, without moving a validated number.
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


class ShrinkageFactors(StateFunction):
    """``b_alpha`` for a pressure-only (dead-oil / immiscible) fluid."""

    def __init__(self, model: Any = None, label: str = "b_\\alpha"):
        super().__init__(model, label)
        self.use_phase_pressures = model is not None and model.fluid.has_pc
        if self.use_phase_pressures:
            self.depends_on("PhasePressures")
        else:
            self.depends_on("pressure", STATE)

    def evaluate_on_domain(self, model: Any, state: Any):
        p_phase = phase_pressures(self, model, state)
        if not self.use_phase_pressures:
            return list(model.fluid.b(p_phase[0]))
        return [model.fluid.b(p_phase[i])[i] for i in range(model.n_phases)]


class BlackOilShrinkageFactors(ShrinkageFactors):
    """``b_alpha`` for a black-oil fluid: adds dissolved/vaporised carry."""

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
            return list(model.fluid.b(p_phase[0], rs, sat, rv, sat_g))
        return [model.fluid.b(p_phase[i], rs, sat, rv, sat_g)[i]
                for i in range(model.n_phases)]


__all__ = ["BlackOilShrinkageFactors", "ShrinkageFactors"]


if __name__ == "__main__":
    import numpy as np

    from pymrst.gridprocessing import cart_grid, compute_geometry
    from pymrst.modules.ad_blackoil.models.generic_black_oil_model import (
        GenericBlackOilModel,
    )
    from pymrst.modules.ad_core.statefunctions.pvt_property_functions import (
        ShrinkageFactors as LegacyShrinkageFactors,
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
        """The pressure-only base reproduces the pre-split ShrinkageFactors
        on a dead-oil (oil-water) fluid, value and Jacobian."""
        model, st = _ow_model()
        new = ShrinkageFactors(model).evaluate_on_domain(model, st)
        # legacy needs its container context; evaluate through the grouping
        model.PVTPropertyFunctions.set_state_function(
            "ShrinkageFactors", LegacyShrinkageFactors(model))
        old = model.get_props(st, "ShrinkageFactors")
        for ph in range(model.n_phases):
            assert np.allclose(value(new[ph]), value(old[ph]))
            assert np.allclose((new[ph] - old[ph]).jac[0].toarray(), 0.0)
        print("ok test_base_matches_legacy_on_oil_water")

    def test_blackoil_subclass_declares_saturated_deps():
        """A black-oil (disgas) model makes the subclass depend on rs+status."""
        model, st = _ow_model()
        # force the fluid to look like a disgas fluid for the dependency check
        model.fluid.disgas = True
        sf = BlackOilShrinkageFactors(model)
        names = {(d.name, d.grouping) for d in sf.dependencies}
        assert ("rs", STATE) in names and ("status", STATE) in names
        model.fluid.disgas = False
        print("ok test_blackoil_subclass_declares_saturated_deps")

    test_base_matches_legacy_on_oil_water()
    test_blackoil_subclass_declares_saturated_deps()
    print("\nAll tests passed!")
