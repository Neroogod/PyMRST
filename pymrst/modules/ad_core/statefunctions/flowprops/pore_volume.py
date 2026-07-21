"""Pore volume (MRST ad-core/statefunctions/flowprops).

Ports MRST's `PoreVolume.m` (static, straight from the operators) and
`BlackOilPoreVolume.m` (multiplied by the rock-compressibility factor
``pvMultR(p)``). The pore volume is the coefficient on every accumulation term,
so its pressure dependence is the rock's contribution to compressibility.

The pre-split PyMRST `PoreVolume` always applied the multiplier; MRST keeps the
static and pressure-dependent forms as separate classes. Both are here; the
black-oil subclass reproduces the pre-split behaviour (`validate_phase340.py`).
"""

from __future__ import annotations

from typing import Any

import numpy as np

from pymrst.modules.ad_core.statefunctions.state_function import (
    STATE,
    StateFunction,
)


class PoreVolume(StateFunction):
    """Static pore volume, taken straight from the model operators."""

    def __init__(self, model: Any = None, label: str = "\\Phi"):
        super().__init__(model, label)
        if model is not None:
            pv = getattr(model, "pv_ref", None)
            if pv is None:
                pv = model.operators.pv
            assert pv is not None, "pore volume (pv) missing from the model"

    def _reference_pv(self, model: Any):
        pv = getattr(model, "pv_ref", None)
        return model.operators.pv if pv is None else pv

    def evaluate_on_domain(self, model: Any, state: Any):
        return self._reference_pv(model)


class BlackOilPoreVolume(PoreVolume):
    """Effective pore volume after the rock-compressibility multiplier."""

    def __init__(self, model: Any = None):
        super().__init__(model)
        self.depends_on("pressure", STATE)

    def evaluate_on_domain(self, model: Any, state: Any):
        pv = self._reference_pv(model)
        return pv * model.fluid.pv_mult(state["pressure"])


__all__ = ["BlackOilPoreVolume", "PoreVolume"]


if __name__ == "__main__":
    from pymrst.gridprocessing import cart_grid, compute_geometry
    from pymrst.modules.ad_blackoil.models.generic_black_oil_model import (
        GenericBlackOilModel,
    )
    from pymrst.modules.ad_core.statefunctions.pvt_property_functions import (
        PoreVolume as LegacyPoreVolume,
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

    import numpy as np

    def test_blackoil_pv_matches_legacy():
        """The compressibility-multiplied pore volume reproduces the pre-split
        PoreVolume exactly (value, and Jacobian when the rock is compressible)."""
        model, st = _ow_model()
        new = BlackOilPoreVolume(model).evaluate_on_domain(model, st)
        model.PVTPropertyFunctions.set_state_function(
            "PoreVolume", LegacyPoreVolume(model))
        old = model.get_props(st, "PoreVolume")
        assert np.allclose(value(new), value(old))
        if hasattr(new, "jac") or hasattr(old, "jac"):
            assert np.allclose((new - old).jac[0].toarray(), 0.0)
        print("ok test_blackoil_pv_matches_legacy")

    def test_static_pv_has_no_pressure_derivative():
        """The static base is a constant - no Jacobian."""
        model, st = _ow_model()
        pv = PoreVolume(model).evaluate_on_domain(model, st)
        assert not hasattr(pv, "jac")
        assert np.allclose(pv, model.pv_ref)
        print("ok test_static_pv_has_no_pressure_derivative")

    test_blackoil_pv_matches_legacy()
    test_static_pv_has_no_pressure_derivative()
    print("\nAll tests passed!")
