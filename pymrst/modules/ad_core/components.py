"""Components: the abstraction that lets new physics be *added* to a model.

Until now PyMRST's conservation equations hardcoded "one component per phase".
That is true for black-oil and false for almost everything else: polymer and
surfactant are species dissolved *in* a phase, a compositional species lives in
two phases at once, and a tracer lives in one but is not it. MRST solves this
by making a **component** a first-class object, and the conservation equations
loop over components rather than phases:

    mass_i   = sum_ph  pv * s_ph * density_i,ph
    flux_i   = sum_ph  upwind(density_i,ph) * v_ph

A component only has to say *how much of itself is in each phase* - its
`component_density` - and everything downstream follows. That is the whole
interface, and it is why adding polymer to MRST does not mean touching the
flow model.

**Densities here are on a surface-volume basis**, not a mass basis as in MRST
(where `ComponentPhaseMass` carries a factor of the surface density). The two
differ by one constant per equation, so they have the same solution, and this
convention keeps every previously validated PyMRST result bit-identical.

The black-oil components are the interesting case: `GasComponent` lives in the
gas phase *and* in the oil phase, because that is what dissolved gas means.
Writing it this way is what makes the framework worth having - the same
conservation equation that handles it also handles a tracer.
"""

from typing import Any, List, Optional

import numpy as np


class Component:
    """One conserved species. Subclasses say where it lives."""

    def __init__(self, name: str):
        self.name = name

    def component_density(self, model, state, b, rs=None,
                          rv=None) -> List[Optional[Any]]:
        """How much of this component sits in a unit volume of each phase.

        `None` for a phase the component is absent from. Surface-volume basis:
        for a phase's own component this is just that phase's `b`.
        """
        raise NotImplementedError

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self.name!r})"


class ImmiscibleComponent(Component):
    """A component confined to one phase - the ordinary case.

    Water in black-oil, and oil when there is no vaporised oil.
    """

    def __init__(self, name: str, phase_index: int):
        super().__init__(name)
        self.phase_index = phase_index

    def component_density(self, model, state, b, rs=None, rv=None):
        out: List[Optional[Any]] = [None] * len(b)
        out[self.phase_index] = b[self.phase_index]
        return out


class GasComponent(Component):
    """Gas: in the gas phase, and dissolved in the oil (black-oil's `rs`).

    The component that justifies the abstraction. Its presence in two phases
    is not a special case in the conservation equation - it is just a density
    that happens to be non-`None` twice.
    """

    def __init__(self, name: str = "gas", gas_index: int = 2,
                 oil_index: int = 1):
        super().__init__(name)
        self.gas_index = gas_index
        self.oil_index = oil_index

    def component_density(self, model, state, b, rs=None, rv=None):
        out: List[Optional[Any]] = [None] * len(b)
        out[self.gas_index] = b[self.gas_index]
        if rs is not None:
            out[self.oil_index] = rs * b[self.oil_index]
        return out


class OilComponent(Component):
    """Oil: in the oil phase, and *vaporised* into the gas (black-oil's `rv`).

    The exact mirror of `GasComponent`, and that symmetry is the point: gas
    dissolves into oil and oil vaporises into gas, and the conservation
    equation does not care which is which. It is what a gas-condensate or
    volatile-oil reservoir needs, and it is the difference between a gas cap
    that carries no liquid and one that gives up condensate as it is produced.
    """

    def __init__(self, name: str = "oil", oil_index: int = 1,
                 gas_index: int = 2):
        super().__init__(name)
        self.oil_index = oil_index
        self.gas_index = gas_index

    def component_density(self, model, state, b, rs=None, rv=None):
        out: List[Optional[Any]] = [None] * len(b)
        out[self.oil_index] = b[self.oil_index]
        if rv is not None:
            out[self.gas_index] = rv * b[self.gas_index]
        return out


class ConcentrationComponent(Component):
    """A species dissolved in a phase at a concentration carried on the state.

    Polymer, surfactant, a tracer: the component's amount is `c` per unit
    surface volume of its carrier phase, and `c` is a primary variable. This is
    the class `ad-eor` is written against, and adding one requires no change to
    the flow model - only to what the model lists as its components.
    """

    def __init__(self, name: str, phase_index: int, state_key: str):
        super().__init__(name)
        self.phase_index = phase_index
        self.state_key = state_key

    def component_density(self, model, state, b, rs=None, rv=None):
        out: List[Optional[Any]] = [None] * len(b)
        i = self.phase_index
        out[i] = state[self.state_key] * b[i]
        return out


def blackoil_components(n_phases: int, disgas: bool,
                        vapoil: bool = False) -> List[Component]:
    """The components of a black-oil model: one per phase, plus the two
    dissolutions - gas into oil (`rs`) and oil into gas (`rv`)."""
    from pymrst.modules.ad_blackoil.components import (
        GasComponent as BlackOilGasComponent,
        OilComponent as BlackOilOilComponent,
    )

    names = ["water", "oil", "gas"][:n_phases]
    comps: List[Component] = [ImmiscibleComponent(names[0], 0)]
    if n_phases > 1:
        comps.append(BlackOilOilComponent("oil", oil_index=1, gas_index=2)
                     if vapoil else ImmiscibleComponent(names[1], 1))
    if n_phases > 2:
        comps.append(BlackOilGasComponent("gas", gas_index=2, oil_index=1)
                     if disgas else ImmiscibleComponent("gas", 2))
    return comps


if __name__ == "__main__":
    class _Fluid:
        n_phases = 2
        disgas = False

    class _Model:
        def __init__(self, nph=2, disgas=False):
            self.fluid = type("F", (), {"n_phases": nph, "disgas": disgas})()

    def test_immiscible_component_lives_in_one_phase():
        """Water is water, and it is only in the water."""
        b = [np.full(3, 1.01), np.full(3, 0.99)]
        c = ImmiscibleComponent("water", 0)
        d = c.component_density(_Model(), {}, b)
        assert d[1] is None, "water is not in the oil"
        assert np.allclose(d[0], 1.01)
        print("ok test_immiscible_component_lives_in_one_phase")

    def test_gas_component_lives_in_two_phases():
        """Dissolved gas is the whole point: one component, two phases."""
        b = [np.full(3, 1.0), np.full(3, 1.2), np.full(3, 0.01)]
        rs = np.full(3, 150.0)
        d = GasComponent().component_density(_Model(3, True), {}, b, rs)
        assert d[0] is None, "no gas in the water"
        assert np.allclose(d[2], 0.01), "free gas"
        assert np.allclose(d[1], 150.0 * 1.2), "and gas dissolved in the oil"
        print("ok test_gas_component_lives_in_two_phases")

    def test_gas_without_disgas_is_just_immiscible():
        """No rs means the gas component is confined to the gas."""
        b = [np.full(2, 1.0), np.full(2, 1.2), np.full(2, 0.01)]
        d = GasComponent().component_density(_Model(3, False), {}, b, rs=None)
        assert d[1] is None and np.allclose(d[2], 0.01)
        print("ok test_gas_without_disgas_is_just_immiscible")

    def test_concentration_component_scales_with_its_variable():
        """A tracer's amount is its concentration times the carrier's b."""
        b = [np.full(3, 1.05), np.full(3, 0.9)]
        c = ConcentrationComponent("tracer", 0, "cT")
        d = c.component_density(_Model(), {"cT": np.array([0.0, 0.5, 1.0])}, b)
        assert d[1] is None
        assert np.allclose(d[0], np.array([0.0, 0.5, 1.0]) * 1.05)
        print("ok test_concentration_component_scales_with_its_variable")

    def test_blackoil_component_sets():
        """Two-phase, three-phase, live oil, and wet gas."""
        assert [type(c).__name__ for c in blackoil_components(2, False)] == \
            ["ImmiscibleComponent"] * 2
        three = blackoil_components(3, True)
        assert type(three[2]).__name__ == "GasComponent"
        assert type(three[1]).__name__ == "ImmiscibleComponent"
        dry = blackoil_components(3, False)
        assert type(dry[2]).__name__ == "ImmiscibleComponent"
        wet = blackoil_components(3, True, vapoil=True)
        assert type(wet[1]).__name__ == "OilComponent"
        assert type(wet[2]).__name__ == "GasComponent"
        print("ok test_blackoil_component_sets")

    def test_oil_component_mirrors_the_gas_one():
        """The whole claim of the abstraction: `rv` is `rs` with the phases
        swapped, and the conservation equation cannot tell them apart."""
        b = [np.full(3, 1.05), np.full(3, 1.2), np.full(3, 0.9)]
        rs = np.array([0.0, 100.0, 200.0])
        rv = np.array([0.0, 5e-4, 1e-3])

        gas = GasComponent("gas", gas_index=2, oil_index=1)
        oil = OilComponent("oil", oil_index=1, gas_index=2)
        dg = gas.component_density(None, {}, b, rs, rv)
        do = oil.component_density(None, {}, b, rs, rv)

        assert dg[0] is None and do[0] is None      # neither is in the water
        # each is `b` in its own phase...
        assert np.allclose(dg[2], b[2]) and np.allclose(do[1], b[1])
        # ...and `r * b` in the other one.
        assert np.allclose(dg[1], rs * b[1])
        assert np.allclose(do[2], rv * b[2])
        print("ok test_oil_component_mirrors_the_gas_one")

    def test_a_dry_gas_carries_no_oil():
        """Without `rv`, the oil component must not leak into the gas phase -
        which is the entire difference between a dry gas cap and a wet one."""
        b = [np.full(2, 1.0), np.full(2, 1.2), np.full(2, 0.9)]
        oil = OilComponent("oil", oil_index=1, gas_index=2)
        d = oil.component_density(None, {}, b, rs=None, rv=None)
        assert d[2] is None, "a dry gas must hold no oil at all"
        assert np.allclose(d[1], b[1])
        print("ok test_a_dry_gas_carries_no_oil")

    test_immiscible_component_lives_in_one_phase()
    test_gas_component_lives_in_two_phases()
    test_gas_without_disgas_is_just_immiscible()
    test_concentration_component_scales_with_its_variable()
    test_blackoil_component_sets()
    test_oil_component_mirrors_the_gas_one()
    test_a_dry_gas_carries_no_oil()
    print("\nAll tests passed!")
