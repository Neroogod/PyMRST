"""MRST ``ad-core/statefunctions/flowprops/BlackOilCapillaryPressure.m``.

Capillary pressure of each phase relative to the reference (oil) pressure:
water carries `-pcOW(sW)`, gas `+pcOG(sG)`, and in a water-gas system gas
carries `pcWG(sG)`. Water saturation can be end-point scaled, and either curve
can be J-function scaled by `sqrt(phi/k)`-style rock ratios (ECLIPSE's
JFUNC). A phase with no capillary pressure gets zeros once any phase has one.
"""

from __future__ import annotations

from typing import Any, List, Optional

import numpy as np

from pymrst.modules.ad_core.ad_oo.statefunctions.flowprops.saturation_property import (
    CONNATE,
    MAXIMUM,
    SaturationProperty,
)
from pymrst.modules.ad_core.statefunctions.state_function import (
    STATE,
    StateFunction,
)
from pymrst.utils.adi import value


class BlackOilCapillaryPressure(StateFunction, SaturationProperty):
    """`pc_alpha` per phase; `None` where every phase shares one pressure."""

    def __init__(self, model: Any, regions: Any = None, *,
                 scalingActive: bool = False):
        StateFunction.__init__(self, model, label="p_{c}", regions=regions)
        self.AutoDiffBackend = model.AutoDiffBackend
        self.scalingActive = scalingActive
        self.surfaceTensionOW: Optional[Any] = None
        self.surfaceTensionOG: Optional[Any] = None
        self.porosityExponent: Optional[float] = None
        self.permeabilityExponent: Optional[float] = None
        self.permeabilityDirection: Optional[Any] = None
        self.pressureUnit = 1.0
        self.depends_on("s", STATE)

    def evaluate_on_domain(self, model: Any, state: Any) -> List[Any]:
        active = model.get_active_phases()
        ph_ind = np.flatnonzero(active) + 1
        nph = int(active.sum())
        pc: List[Any] = [None]*nph
        f = model.fluid
        jfunc_ow = self.has_j_function_scaler("OW")
        jfunc_og = self.has_j_function_scaler("OG")
        ratio = self.get_j_function_static_ratio(model) if (jfunc_ow or jfunc_og) else None
        has_pc = False
        if model.water and model.oil and getattr(f, "pcOW", None) is not None:
            sw = model.get_prop(state, "sw")
            if self.scalingActive:
                pts = model.rock.krscale.drainage
                get = SaturationProperty.get_sat_point_picker(
                    f, pts, self.regions, self.cell_subset)
                swcon, SWCON = get("w", CONNATE)
                swmax, SWMAX = get("w", MAXIMUM)
                sw = swcon + (sw - SWCON)*(swmax - swcon)/(SWMAX - SWCON)
            pcow = self.evaluate_fluid(model, "pcOW", sw)
            if "pcowScale" in state:
                assert not jfunc_ow, (
                    "Cannot both have initial water pc scale and Jfunc scaling.")
                pcow = pcow*state["pcowScale"]
            elif jfunc_ow:
                pcow = self.surfaceTensionOW*ratio*pcow
            pc[int(np.flatnonzero(ph_ind == 1)[0])] = -pcow
            has_pc = True
        if model.gas and model.oil and getattr(f, "pcOG", None) is not None:
            sg = model.get_prop(state, "sg")
            pcog = self.evaluate_fluid(model, "pcOG", sg)
            if jfunc_og:
                pcog = self.surfaceTensionOG*ratio*pcog
            pc[int(np.flatnonzero(ph_ind == 3)[0])] = pcog
            has_pc = True
        if not model.oil and getattr(f, "pcWG", None) is not None:
            sg = model.get_prop(state, "sg")
            pc[int(np.flatnonzero(ph_ind == 3)[0])] = self.evaluate_fluid(
                model, "pcWG", sg)
            has_pc = True
        if has_pc:
            n = max(np.size(value(x)) for x in pc if x is not None)
            pc = [np.zeros(n) if x is None else x for x in pc]
        return pc

    def pc_present(self, model: Any) -> bool:
        f = model.fluid
        return any(getattr(f, name, None) is not None
                   for name in ("pcOW", "pcOG", "pcWG"))

    def subset(self, subs: Any) -> "BlackOilCapillaryPressure":
        out = super().subset(subs)
        out.cell_subset = subs
        return out

    def set_j_function_constants(self, poroexp: float, permexp: float,
                                 permdir: Any, pressureunit: float = None):
        if pressureunit is not None:
            self.pressureUnit = pressureunit
        self.porosityExponent = poroexp
        self.permeabilityExponent = permexp
        if isinstance(permdir, str):
            permdir = [("xyz".index(c.lower())) for c in permdir]
        self.permeabilityDirection = permdir
        return self

    def set_surface_tension(self, surface_tension: Any, fluidpair: str):
        pair = fluidpair.lower()
        if pair == "ow":
            self.surfaceTensionOW = surface_tension
        elif pair == "og":
            self.surfaceTensionOG = surface_tension
        else:
            raise ValueError(f"Unsupported pair {fluidpair}")
        return self

    def has_j_function_scaler(self, phasepair: str) -> bool:
        return self.scalingActive and self.get_surface_tension(phasepair) is not None

    def get_surface_tension(self, phasepair: str) -> Any:
        return getattr(self, f"surfaceTension{phasepair.upper()}")

    def get_j_function_static_ratio(self, model: Any) -> np.ndarray:
        """`phi^a / k^b`, divided by the table's pressure unit."""
        cells = (slice(None) if isinstance(self.cell_subset, str)
                 else self.cell_subset)
        phi = np.asarray(model.rock.poro, dtype=float).ravel()[cells]
        perm = np.atleast_2d(np.asarray(model.rock.perm, dtype=float).T).T
        k = perm[cells][:, self.permeabilityDirection]
        k = np.sum(k, axis=1)/k.shape[1]
        ratio = phi**self.porosityExponent/k**self.permeabilityExponent
        return ratio/self.pressureUnit

    evaluateOnDomain = evaluate_on_domain
    pcPresent = pc_present
    setJFunctionConstants = set_j_function_constants
    setSurfaceTension = set_surface_tension
    hasJFunctionScaler = has_j_function_scaler
    getSurfaceTension = get_surface_tension
    getJFunctionStaticRatio = get_j_function_static_ratio


__all__ = ["BlackOilCapillaryPressure"]


if __name__ == "__main__":
    from pymrst.modules.ad_core.ad_oo.testing import toy_model, toy_state

    def test_no_curves_means_no_capillary_pressure():
        model = toy_model("WO", nx=2)
        pc = model.get_prop(toy_state(model), "CapillaryPressure")
        assert pc == [None, None]
        print("ok test_no_curves_means_no_capillary_pressure")

    def test_water_carries_minus_pcow_and_oil_gets_zeros():
        model = toy_model("WO", nx=2, pc_ow=lambda sw: 1e5*(1.0 - sw))
        state = toy_state(model, s=np.array([[0.2, 0.8], [0.6, 0.4]]))
        pc_w, pc_o = model.get_prop(state, "CapillaryPressure")
        np.testing.assert_allclose(pc_w, -1e5*np.array([0.8, 0.4]))
        np.testing.assert_allclose(pc_o, 0.0)
        print("ok test_water_carries_minus_pcow_and_oil_gets_zeros")

    def test_gas_carries_plus_pcog():
        model = toy_model("WOG", nx=1, pc_og=lambda sg: 2e5*sg)
        state = toy_state(model, s=np.array([[0.2, 0.5, 0.3]]))
        pc = model.get_prop(state, "CapillaryPressure")
        np.testing.assert_allclose(pc[2], 0.6e5)
        print("ok test_gas_carries_plus_pcog")

    test_no_curves_means_no_capillary_pressure()
    test_water_carries_minus_pcow_and_oil_gets_zeros()
    test_gas_carries_plus_pcog()
    print("\nAll tests passed!")
