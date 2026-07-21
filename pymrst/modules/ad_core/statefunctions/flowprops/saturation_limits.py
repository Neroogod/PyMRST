"""Saturated-ratio limits (MRST ad-core/statefunctions/flowprops).

Ports MRST's `RsMax.m` and `RvMax.m`: the maximum dissolved gas-oil ratio and
maximum vaporised oil-gas ratio a cell can hold at its pressure - the
bubble-point / dew-point envelopes. A cell is *saturated* when its ``rs``
(resp. ``rv``) sits on this curve, and the hydrocarbon-status switch that
`GenericBlackOilModel` resolves is exactly the comparison of ``rs`` to
``RsMax``.

``RsMax`` is a function of cell pressure; ``RvMax`` of the gas-phase pressure
(the two differ once capillary pressure is active). Both read the saturation
tables off ``model.fluid`` (``rs_sat`` / ``rv_sat``).
"""

from __future__ import annotations

from typing import Any

from pymrst.modules.ad_core.statefunctions.state_function import (
    STATE,
    StateFunction,
)


class RsMax(StateFunction):
    """Maximum dissolved gas-oil ratio ``rsSat(p)`` (bubble-point)."""

    def __init__(self, model: Any = None, label: str = "R_s^{max}"):
        super().__init__(model, label)
        self.depends_on("pressure", STATE)

    def evaluate_on_domain(self, model: Any, state: Any):
        return model.fluid.rs_sat(state["pressure"])


class RvMax(StateFunction):
    """Maximum vaporised oil-gas ratio ``rvSat(p_gas)`` (dew-point)."""

    def __init__(self, model: Any = None, label: str = "R_v^{max}"):
        super().__init__(model, label)
        self.depends_on("PhasePressures")

    def evaluate_on_domain(self, model: Any, state: Any):
        p_phase = self.get_evaluated(state, "PhasePressures")
        # gas is the last phase (water + oil + gas ordering)
        pg = p_phase[model.n_phases - 1]
        return model.fluid.rv_sat(pg)


__all__ = ["RsMax", "RvMax"]


if __name__ == "__main__":
    import numpy as np

    from pymrst.gridprocessing import cart_grid, compute_geometry
    from pymrst.modules.ad_blackoil.models.generic_black_oil_model import (
        GenericBlackOilModel,
    )
    from pymrst.modules.ad_core.statefunctions.state_function import State
    from pymrst.modules.ad_props import init_simple_adi_fluid
    from pymrst.params import add_well, make_rock
    from pymrst.utils.adi import value
    from pymrst.utils.units import BARSA, CENTI, DARCY, MILLI, POISE

    def _model():
        g = compute_geometry(cart_grid([5, 4, 1], [250.0, 200.0, 10.0]))
        rock = make_rock(g, 100 * MILLI * DARCY, 0.25)
        fluid = init_simple_adi_fluid(
            mu=np.array([1.0, 5.0]) * CENTI * POISE, rho=[1000.0, 800.0],
            n=[2, 2], c=[1e-6 / BARSA, 1e-4 / BARSA], p_ref=200 * BARSA)
        # attach a synthetic linear rs_sat so RsMax has something to read
        fluid_rs = 1e-4
        model_fluid = fluid
        w = add_well(None, g, rock, np.array([0]), well_type="rate", val=2e-3,
                     compi=np.array([1.0, 0.0]), sign=1)
        model = GenericBlackOilModel(g, rock, fluid, w)
        model.fluid.rs_sat = lambda p: fluid_rs * value(p)
        st = model.init_state_ad(State(
            pressure=np.linspace(210, 190, g.cells.num) * BARSA,
            sW=np.full(g.cells.num, 0.5), qWs=np.array([2e-3]),
            qOs=np.array([-2e-3]), bhp=np.array([150 * BARSA])))
        return model, st

    def test_rsmax_reads_saturation_table():
        """RsMax returns fluid.rs_sat evaluated at cell pressure."""
        model, st = _model()
        rsmax = RsMax(model).evaluate_on_domain(model, st)
        assert np.allclose(value(rsmax),
                           value(model.fluid.rs_sat(st["pressure"])))
        print("ok test_rsmax_reads_saturation_table")

    def test_rsmax_depends_on_pressure():
        model, st = _model()
        deps = {(d.name, d.grouping) for d in RsMax(model).dependencies}
        assert ("pressure", STATE) in deps
        print("ok test_rsmax_depends_on_pressure")

    test_rsmax_reads_saturation_table()
    test_rsmax_depends_on_pressure()
    print("\nAll tests passed!")
