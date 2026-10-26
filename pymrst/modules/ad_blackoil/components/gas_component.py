"""MRST ``ad-blackoil/components/GasComponent.m``.

`GasComponent` is MRST's class for the AD-OO generic black-oil model.
`GasComponentAD` is PyMRST's earlier component for the legacy models
(`blackoil_components`), kept for them.
"""

from typing import Any, List, Optional

from pymrst.modules.ad_core.ad_oo.statefunctions.components.immiscible_component import (
    ImmiscibleComponent,
)
from pymrst.modules.ad_core.statefunctions.components.components import Component


class GasComponent(ImmiscibleComponent):
    """MRST `GasComponent`: gas in the gas phase, and dissolved in the oil
    phase through `rs` when the model has dissolved gas."""

    def __init__(self, name: str, phase_index: int, disgas: bool, vapoil: bool):
        super().__init__(name, phase_index)
        self.disgas = bool(disgas)
        self.vapoil = bool(vapoil)
        self.function_depends_on("getComponentDensity", "ShrinkageFactors",
                                 "PVTPropertyFunctions")
        if self.disgas:
            self.function_depends_on("getComponentDensity", "rs", "state")

    def get_component_density(self, model: Any, state: Any, *args: Any):
        """Surface-gas mass per reservoir volume in each phase."""
        if not (self.disgas or self.vapoil):
            return super().get_component_density(model, state, *args)
        names = model.get_phase_names()
        c: List[Any] = [None]*len(names)
        pvt = model.PVTPropertyFunctions
        gix = names.index("G")
        rho = pvt.get(model, state, "Density")
        b = pvt.get(model, state, "ShrinkageFactors")
        rho_s = pvt.get(model, state, "SurfaceDensity")
        rho_gs = rho_s[gix]
        c[gix] = rho_gs*b[gix] if self.vapoil else rho[gix]
        if self.disgas:
            oix = names.index("O")
            rs = model.get_prop(state, "rs")
            c[oix] = rs*rho_gs*b[oix]
        return c

    getComponentDensity = get_component_density


class GasComponentAD(Component):
    """Gas: in the gas phase, and dissolved in the oil through `rs`."""

    def __init__(self, name: str = "gas", gas_index: int = 2,
                 oil_index: int = 1):
        super().__init__(name)
        self.gas_index = gas_index
        self.oil_index = oil_index

    def component_density(self, model, state, b, rs=None,
                          rv=None) -> List[Optional[Any]]:
        out: List[Optional[Any]] = [None] * len(b)
        out[self.gas_index] = b[self.gas_index]
        if rs is not None:
            out[self.oil_index] = rs * b[self.oil_index]
        return out


    def get_phase_composition(self, model, state=None, *args):
        """The fraction of each phase made up of this component.

        `1` in the gas phase, `None` elsewhere.

        `None` rather than `0` is MRST's distinction and it is load-bearing:
        the conservation equation skips a `None` slot, while a `0` contributes
        a term that is identically zero and still carries an AD derivative and
        a sparsity entry.

        Not the oil phase, even under `disgas` - and that is not an oversight.
        It is the difference between *composition* and *density*: the oil
        phase does carry dissolved gas, and `component_density` reports it
        as `rs * b_o`, but the oil phase is not made *of* gas. MRST's
        `GasComponent.getPhaseComposition` delegates straight to the
        immiscible base for exactly this reason.
        """
        from pymrst.modules.ad_core.statefunctions.components.components import (
            _phase_composition,
        )
        return _phase_composition(model, self.gas_index)

    getPhaseComposition = get_phase_composition

__all__ = ["GasComponent", "GasComponentAD"]


if __name__ == "__main__":
    import numpy as np

    from pymrst.gridprocessing import cart_grid, compute_geometry
    from pymrst.modules.ad_blackoil.components.oil_component import OilComponent
    from pymrst.modules.ad_blackoil.models import GenericBlackOilModel
    from pymrst.modules.test_suite.dataset_functions.spe1 import setup_spe1
    from pymrst.params.gravity.gravity import isolated_gravity
    from pymrst.params.rock import make_rock

    def live_oil_model_and_state():
        _, _, fluid, _, _ = setup_spe1()
        G = compute_geometry(cart_grid([2, 1, 1]))
        model = GenericBlackOilModel(G, make_rock(G, 1e-13, 0.3), fluid,
                                     disgas=True).validate_model()
        state = {"pressure": np.array([200e5, 250e5]),
                 "s": np.array([[0.2, 0.8, 0.0], [0.2, 0.6, 0.2]]),
                 "rs": np.array([100.0, 150.0])}
        return model, model.init_state_function_containers(model.validate_state(state))

    def test_dissolved_gas_rides_in_the_oil_phase():
        with isolated_gravity():
            model, state = live_oil_model_and_state()
            c = GasComponent("gas", 2, True, False).get_component_density(model, state)
            rho_s = model.get_prop(state, "SurfaceDensity")
            b = model.get_prop(state, "ShrinkageFactors")
            rho = model.get_prop(state, "Density")
        np.testing.assert_allclose(c[1], state["rs"]*rho_s[2]*b[1], rtol=1e-14)
        np.testing.assert_allclose(c[2], rho[2], rtol=1e-14)
        print("ok test_dissolved_gas_rides_in_the_oil_phase")

    def test_oil_and_gas_components_add_up_to_the_oil_density():
        with isolated_gravity():
            model, state = live_oil_model_and_state()
            oil = OilComponent("oil", 1, True, False).get_component_density(model, state)
            gas = GasComponent("gas", 2, True, False).get_component_density(model, state)
            rho = model.get_prop(state, "Density")
        np.testing.assert_allclose(oil[1] + gas[1], rho[1], rtol=1e-12)
        print("ok test_oil_and_gas_components_add_up_to_the_oil_density")

    test_dissolved_gas_rides_in_the_oil_phase()
    test_oil_and_gas_components_add_up_to_the_oil_density()
    print("\nAll tests passed!")
