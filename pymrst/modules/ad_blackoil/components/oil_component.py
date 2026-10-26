"""MRST ``ad-blackoil/components/OilComponent.m``.

`OilComponent` is MRST's class for the AD-OO generic black-oil model.
`OilComponentAD` is PyMRST's earlier component for the legacy models
(`blackoil_components`), kept for them.
"""

from typing import Any, List, Optional

from pymrst.modules.ad_core.ad_oo.statefunctions.components.immiscible_component import (
    ImmiscibleComponent,
)
from pymrst.modules.ad_core.statefunctions.components.components import Component


class OilComponent(ImmiscibleComponent):
    """MRST `OilComponent`: oil in the oil phase, and vaporised in the gas
    phase through `rv` when the model has vaporised oil."""

    def __init__(self, name: str, phase_index: int, disgas: bool, vapoil: bool):
        super().__init__(name, phase_index)
        self.disgas = bool(disgas)
        self.vapoil = bool(vapoil)
        self.function_depends_on("getComponentDensity",
                                 ["ShrinkageFactors", "SurfaceDensity"],
                                 "PVTPropertyFunctions")
        if self.vapoil:
            self.function_depends_on("getComponentDensity", "rv", "state")

    def get_component_density(self, model: Any, state: Any, *args: Any):
        """Surface-oil mass per reservoir volume in each phase."""
        if not (self.disgas or self.vapoil):
            return super().get_component_density(model, state, *args)
        names = model.get_phase_names()
        c: List[Any] = [None]*len(names)
        pvt = model.PVTPropertyFunctions
        rho_s = pvt.get(model, state, "SurfaceDensity")
        rho = pvt.get(model, state, "Density")
        b = pvt.get(model, state, "ShrinkageFactors")
        oix = names.index("O")
        rho_os = rho_s[oix]
        c[oix] = rho_os*b[oix] if self.disgas else rho[oix]
        if self.vapoil:
            gix = names.index("G")
            rv = model.get_prop(state, "rv")
            c[gix] = rv*rho_os*b[gix]
        return c

    getComponentDensity = get_component_density


class OilComponentAD(Component):
    """Oil: in the oil phase, and vaporised into the gas through `rv`."""

    def __init__(self, name: str = "oil", oil_index: int = 1,
                 gas_index: int = 2):
        super().__init__(name)
        self.oil_index = oil_index
        self.gas_index = gas_index

    def component_density(self, model, state, b, rs=None,
                          rv=None) -> List[Optional[Any]]:
        out: List[Optional[Any]] = [None] * len(b)
        out[self.oil_index] = b[self.oil_index]
        if rv is not None:
            out[self.gas_index] = rv * b[self.gas_index]
        return out


    def get_phase_composition(self, model, state=None, *args):
        """The fraction of each phase made up of this component.

        `1` in the oil phase, `None` elsewhere.

        `None` rather than `0` is MRST's distinction and it is load-bearing:
        the conservation equation skips a `None` slot, while a `0` contributes
        a term that is identically zero and still carries an AD derivative and
        a sparsity entry.

        The mirror of `GasComponent`'s: the gas phase carries vaporised oil
        through `rv`, but its composition is gas.
        """
        from pymrst.modules.ad_core.statefunctions.components.components import (
            _phase_composition,
        )
        return _phase_composition(model, self.oil_index)

    getPhaseComposition = get_phase_composition

__all__ = ["OilComponent", "OilComponentAD"]


if __name__ == "__main__":
    import numpy as np

    from pymrst.gridprocessing import cart_grid, compute_geometry
    from pymrst.modules.ad_blackoil.models import GenericBlackOilModel
    from pymrst.modules.test_suite.dataset_functions.spe1 import setup_spe1
    from pymrst.params.gravity.gravity import isolated_gravity
    from pymrst.params.rock import make_rock

    def live_oil_model_and_state():
        """SPE1's live oil; `rv` is set by hand, as SPE1's gas is dry."""
        _, _, fluid, _, _ = setup_spe1()
        G = compute_geometry(cart_grid([2, 1, 1]))
        model = GenericBlackOilModel(G, make_rock(G, 1e-13, 0.3), fluid,
                                     disgas=True).validate_model()
        state = {"pressure": np.array([200e5, 250e5]),
                 "s": np.array([[0.2, 0.8, 0.0], [0.2, 0.6, 0.2]]),
                 "rs": np.array([100.0, 150.0]), "rv": np.array([1e-4, 2e-4])}
        return model, model.init_state_function_containers(model.validate_state(state))

    def test_live_oil_carries_surface_oil_density_times_b():
        with isolated_gravity():
            model, state = live_oil_model_and_state()
            c = OilComponent("oil", 1, True, False).get_component_density(model, state)
            rho_s = model.get_prop(state, "SurfaceDensity")
            b = model.get_prop(state, "ShrinkageFactors")
        np.testing.assert_allclose(c[1], rho_s[1]*b[1], rtol=1e-14)
        assert c[0] is None and c[2] is None
        print("ok test_live_oil_carries_surface_oil_density_times_b")

    def test_dead_oil_density_is_the_phase_density():
        with isolated_gravity():
            model, state = live_oil_model_and_state()
            c = OilComponent("oil", 1, False, False).get_component_density(model, state)
            rho = model.get_prop(state, "Density")
        np.testing.assert_allclose(c[1], rho[1], rtol=1e-14)
        print("ok test_dead_oil_density_is_the_phase_density")

    def test_vaporised_oil_rides_in_the_gas_phase():
        with isolated_gravity():
            model, state = live_oil_model_and_state()
            component = OilComponent("oil", 1, True, True)
            c = component.get_component_density(model, state)
            rho_s = model.get_prop(state, "SurfaceDensity")
            b = model.get_prop(state, "ShrinkageFactors")
            rv = model.get_prop(state, "rv")
        np.testing.assert_allclose(c[2], rv*rho_s[1]*b[2], rtol=1e-14)
        assert "rv" in [d.name for d in
                        component.get_function_dependencies("getComponentDensity")["externals"]]
        print("ok test_vaporised_oil_rides_in_the_gas_phase")

    test_live_oil_carries_surface_oil_density_times_b()
    test_dead_oil_density_is_the_phase_density()
    test_vaporised_oil_rides_in_the_gas_phase()
    print("\nAll tests passed!")
