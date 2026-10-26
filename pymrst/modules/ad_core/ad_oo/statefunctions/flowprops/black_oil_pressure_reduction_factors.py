"""MRST ``ad-core/statefunctions/flowprops/BlackOilPressureReductionFactors.m``.

True-IMPES weights for a black-oil fluid. Combining the phase equations with
these weights cancels the saturation derivatives, leaving a pressure equation;
with dissolution the weights mix the oil and gas shrinkage factors through
`rs` and `rv`, and in undersaturated cells they also need `dbO/drs`.
"""

from __future__ import annotations

from typing import Any, List

import numpy as np

from pymrst.modules.ad_core.ad_oo.statefunctions.flowprops.pressure_reduction_factors import (
    PressureReductionFactors,
)
from pymrst.modules.ad_core.statefunctions.state_function import STATE
from pymrst.utils.adi import init_variables_adi, value


class BlackOilPressureReductionFactors(PressureReductionFactors):
    """`w_alpha^p` for live oil / wet gas, divided by pore volume."""

    def __init__(self, model: Any, regions: Any = None):
        super().__init__(model, regions)
        from pymrst.modules.ad_blackoil.models.three_phase_black_oil_model import (
            ThreePhaseBlackOilModel,
        )
        assert isinstance(model, ThreePhaseBlackOilModel), (
            "Model must be derived from the black-oil model. Did you want the "
            "regular PressureReductionFactors class instead?")
        self.reset_dependencies()
        self.vapoil = bool(model.vapoil)
        self.disgas = bool(model.disgas)
        self.useUndersaturated = None
        if self.disgas:
            self.depends_on("rs", STATE)
            if not self.vapoil and self.useUndersaturated is None:
                self.useUndersaturated = True
        elif self.useUndersaturated is None:
            self.useUndersaturated = False
        if self.vapoil:
            self.depends_on("rv", STATE)
        self.depends_on(["ShrinkageFactors", "PoreVolume", "Density"])

    def evaluate_on_domain(self, model: Any, state: Any) -> List[Any]:
        b, pv, rho = self.get_evaluated_dependencies(
            state, "ShrinkageFactors", "PoreVolume", "Density")
        rho_s = model.get_surface_densities()
        vap, dis = self.vapoil, self.disgas
        oix = gix = None
        if vap or dis:
            oix, gix = model.get_phase_index("O", "G")
        rv = model.get_prop(state, "rv") if vap else 0.0
        do_usat = False
        if dis:
            rs = model.get_prop(state, "rs")
            do_usat = bool(self.useUndersaturated)
            if do_usat:
                assert not model.vapoil
                bo = b[oix]
                sg = model.get_prop(state, "sg")
                p = model.get_prop(state, "pressure")
                usat = np.asarray(value(sg)) == 0
                (rs_ad,) = init_variables_adi(np.asarray(value(rs), dtype=float))
                shrinkage = model.PVTPropertyFunctions.get_state_function(
                    "ShrinkageFactors")
                tmp = shrinkage.evaluate_function_on_domain_with_arguments(
                    model.fluid.bO, np.asarray(value(p), dtype=float), rs_ad,
                    np.zeros(np.size(value(rs)), dtype=bool))
                if np.any(usat) and hasattr(tmp, "jac"):
                    dbo_drsu = tmp.jac[0].diagonal()[usat]
                    bou = bo[usat]
                    rsu = rs[usat]
                else:
                    do_usat = False
        else:
            rs = 0.0
        alpha = 1.0/(1.0 - dis*vap*rs*rv)
        names = model.get_component_names() or model.get_long_phase_names()
        pvt_regions = model.PVTPropertyFunctions.get_state_function("Density").regions
        w: List[Any] = []
        for ph in range(len(b)):
            name = names[ph]
            if name == "water":
                f = 1.0/rho[ph]
            elif name == "oil":
                if dis:
                    rho_os = rho_s[np.asarray(pvt_regions) - 1, ph]
                    f = (alpha/rho_os)*(1.0/b[oix] - dis*rs/b[gix])
                    if do_usat:
                        f[usat] = (1.0/(rho_os[usat]*bou))*(1.0 + (rsu/bou)*dbo_drsu)
                else:
                    f = 1.0/rho[ph]
            elif name == "gas":
                rho_gs = rho_s[np.asarray(pvt_regions) - 1, ph]
                if vap:
                    f = (alpha/rho_gs)*(1.0/b[ph] - vap*rv/b[oix])
                else:
                    f = 1.0/rho[ph]
                    if do_usat:
                        f[usat] = -(1.0/(rho_gs[usat]*bou**2))*dbo_drsu
            else:
                f = 0.0
            w.append(f/pv)
        return w

    evaluateOnDomain = evaluate_on_domain


__all__ = ["BlackOilPressureReductionFactors"]


if __name__ == "__main__":
    def test_a_model_that_is_not_black_oil_is_refused():
        from pymrst.modules.ad_core.ad_oo.testing import toy_model
        try:
            BlackOilPressureReductionFactors(toy_model("WOG", nx=2))
            raise SystemExit("should have raised")
        except AssertionError:
            pass
        print("ok test_a_model_that_is_not_black_oil_is_refused")

    test_a_model_that_is_not_black_oil_is_refused()
    print("\nAll tests passed!")
