"""Pressure-reduction factors (MRST ad-core/statefunctions/flowprops).

Ports MRST's `PressureReductionFactors.m` (immiscible base) and
`BlackOilPressureReductionFactors.m` (the live-oil / wet-gas weights).

These are the per-component weights ``w_i`` that collapse the component
conservation system to a single pressure equation: the weighted sum
``sum_i w_i M_i`` of the component masses is a *volume* balance, so its
derivative with respect to the saturation/composition unknowns (nearly)
vanishes and what is left is driven by pressure. This is the "true-IMPES"
weighting a CPR pressure stage and the sequential pressure solve are built on.

For the immiscible base each phase is its own component and
``w_alpha = pv / rho_alpha``. For black-oil the dissolved-gas / vaporised-oil
cross terms couple oil and gas, giving the ``1/b_o - rs/b_g`` style weights
below. The defining invariant - ``sum_i w_i M_i`` has no saturation derivative -
is what `validate_phase341.py` checks, because there is no simpler reference
for machinery whose whole purpose is to be saturation-independent.

Not ported here: `NumericalPressureReductionFactors` (weights differentiated
numerically from the mass Jacobian - tied to the diagonal-AD backend and the
sequential solver's `reductionFactorProps`), and the ``useUndersaturated``
diagonal-AD refinement of the black-oil weights. Both are accuracy/plumbing
refinements on top of the analytic weights implemented here.
"""

from __future__ import annotations

from typing import Any, List

import numpy as np

from pymrst.modules.ad_core.statefunctions.state_function import (
    STATE,
    StateFunction,
)


class PressureReductionFactors(StateFunction):
    """Immiscible pressure weights: ``w_alpha = pv / rho_alpha``."""

    def __init__(self, model: Any = None, label: str = "w_i^p"):
        super().__init__(model, label)
        # In PyMRST the PVT properties live in the PVTPropertyFunctions
        # grouping (MRST files them under flowprops too); depend externally.
        self.depends_on(["Density", "PoreVolume"], "PVTPropertyFunctions")

    def evaluate_on_domain(self, model: Any, state: Any) -> List[Any]:
        rho = self.get_evaluated_external(
            model, state, "PVTPropertyFunctions", "Density")
        pv = self.get_evaluated_external(
            model, state, "PVTPropertyFunctions", "PoreVolume")
        return [pv / rho[ph] for ph in range(len(rho))]


class BlackOilPressureReductionFactors(PressureReductionFactors):
    """Black-oil pressure weights with dissolved-gas / vaporised-oil coupling."""

    def __init__(self, model: Any = None):
        super().__init__(model)
        self.dependencies = []                    # MRST resetDependencies()
        self.disgas = model is not None and model.fluid.disgas
        self.vapoil = model is not None and model.fluid.vapoil
        if self.disgas:
            self.depends_on("rs", STATE)
        if self.vapoil:
            self.depends_on("rv", STATE)
        self.depends_on(["ShrinkageFactors", "PoreVolume", "Density"],
                        "PVTPropertyFunctions")

    def evaluate_on_domain(self, model: Any, state: Any) -> List[Any]:
        ext = self.get_evaluated_external
        b = ext(model, state, "PVTPropertyFunctions", "ShrinkageFactors")
        pv = ext(model, state, "PVTPropertyFunctions", "PoreVolume")
        rho = ext(model, state, "PVTPropertyFunctions", "Density")
        rho_s = np.asarray(model.fluid.rho_s, dtype=float)
        nph = len(b)
        oix, gix = 1, 2                            # water, oil, gas ordering

        rs = state["rs"] if self.disgas else 0.0
        rv = state["rv"] if self.vapoil else 0.0
        alpha = 1.0 / (1.0 - (rs * rv if (self.disgas and self.vapoil) else 0.0))

        w: List[Any] = [None] * nph
        for ph in range(nph):
            if ph == 0:                            # water
                f = 1.0 / rho[ph]
            elif ph == oix:                        # oil
                if self.disgas:
                    rho_os = rho_s[oix]
                    f = (alpha / rho_os) * (1.0 / b[oix] - rs / b[gix])
                else:
                    f = 1.0 / rho[ph]
            elif ph == gix:                        # gas
                if self.vapoil:
                    rho_gs = rho_s[gix]
                    f = (alpha / rho_gs) * (1.0 / b[gix] - rv / b[oix])
                else:
                    f = 1.0 / rho[ph]
            else:
                f = 0.0
            w[ph] = f / pv
        return w


__all__ = ["BlackOilPressureReductionFactors", "PressureReductionFactors"]


if __name__ == "__main__":
    from pymrst.gridprocessing import cart_grid, compute_geometry
    from pymrst.modules.ad_blackoil.models.generic_black_oil_model import (
        GenericBlackOilModel,
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

    def test_immiscible_weights_are_pv_over_rho():
        """The base weight is exactly pv / rho per phase."""
        model, st = _ow_model()
        model.FlowPropertyFunctions.set_state_function(
            "PressureReductionFactors", PressureReductionFactors(model))
        w = model.get_props(st, "PressureReductionFactors")
        rho = model.get_props(st, "Density")
        pv = model.get_props(st, "PoreVolume")
        for ph in range(model.n_phases):
            assert np.allclose(value(w[ph]), value(pv) / value(rho[ph]))
        print("ok test_immiscible_weights_are_pv_over_rho")

    def test_true_impes_invariant_immiscible():
        """sum_ph w_ph * (phase mass) has no derivative w.r.t. saturation - the
        defining property of the pressure weight.

        The weight ``w = pv/rho`` is the true-IMPES weight for the *phase mass*
        ``M_ph = pv * s_ph * rho_ph``, so ``sum_ph w_ph M_ph = pv^2 * sum(s) =
        pv^2`` is saturation-independent (the phase saturations sum to one, and
        the graph's Saturations carries that constraint through the Jacobian)."""
        model, st = _ow_model()
        model.FlowPropertyFunctions.set_state_function(
            "PressureReductionFactors", PressureReductionFactors(model))
        w = model.get_props(st, "PressureReductionFactors")
        s = model.get_props(st, "Saturations")
        rho = model.get_props(st, "Density")
        pv = model.get_props(st, "PoreVolume")
        total = sum(w[ph] * (pv * s[ph] * rho[ph])
                    for ph in range(model.n_phases))
        sat_deriv = np.abs(total.jac[1].toarray()).max()   # sW block
        magnitude = np.abs(value(total)).max()
        print(f"  true-IMPES: |d(sum w*M)/dsW| = {sat_deriv:.2e} "
              f"(|sum w*M| ~ {magnitude:.2e})")
        assert sat_deriv < 1e-6 * magnitude
        print("ok test_true_impes_invariant_immiscible")

    test_immiscible_weights_are_pv_over_rho()
    test_true_impes_invariant_immiscible()
    print("\nAll tests passed!")
