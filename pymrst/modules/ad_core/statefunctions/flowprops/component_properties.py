"""Component flow properties (MRST ad-core/statefunctions/flowprops).

Ports MRST's `ComponentProperty` (the virtual base that reads a named method
off each of the model's components), `ComponentPhaseDensity`
(``rho x_{i,alpha}`` - the density of component *i* in phase *alpha*) and
`ComponentMobility`  (``lambda_{i,alpha}`` - the mobile mass of component *i*
in phase *alpha*, i.e. phase mobility times component density).

These are what make a component's physics *composable*: the flux group upwinds
`ComponentMobility` (see `FaceComponentMobility`) and the accumulation reads
`ComponentPhaseDensity`, so adding a species - a tracer, polymer, a dissolved
gas - is a new component object, not a new residual. They iterate over
`model.components`, calling the component's own `get_component_density` /
`get_component_mobility` (the MRST `GenericComponent` method surface, now on
PyMRST's `Component` base).
"""

from __future__ import annotations

from typing import Any, List, Optional

from pymrst.modules.ad_core.statefunctions.state_function import (
    STATE,
    StateFunction,
)


class ComponentProperty:
    """Mixin for a state function whose values come from the components.

    Holds the name of the component method to call (``get_component_density``,
    ``get_component_mobility``) and, on construction, declares the graph
    dependencies those methods reach for so the grouping evaluates them first.
    """

    def __init__(self, model: Any, component_function_name: str):
        self.component_function_name = component_function_name


class ComponentPhaseDensity(StateFunction, ComponentProperty):
    """``rho x_{i,alpha}``: density of each component in each phase, per cell."""

    def __init__(self, model: Any = None):
        StateFunction.__init__(self, model, label="\\rho_\\alpha x_{i,\\alpha}")
        ComponentProperty.__init__(self, model, "get_component_density")
        self.depends_on("ShrinkageFactors", "PVTPropertyFunctions")
        if model is not None and model.fluid.disgas:
            self.depends_on("rs", STATE)
        if model is not None and model.fluid.vapoil:
            self.depends_on("rv", STATE)

    def evaluate_on_domain(self, model: Any, state: Any) -> List[List[Optional[Any]]]:
        return [c.get_component_density(model, state) for c in model.components]


class ComponentMobility(StateFunction, ComponentProperty):
    """``lambda_{i,alpha}``: mobile mass of each component in each phase."""

    def __init__(self, model: Any = None):
        StateFunction.__init__(self, model, label="\\lambda_{i,\\alpha}")
        ComponentProperty.__init__(self, model, "get_component_mobility")
        self.depends_on("Mobility", "FlowPropertyFunctions")
        self.depends_on("ShrinkageFactors", "PVTPropertyFunctions")
        if model is not None and model.fluid.disgas:
            self.depends_on("rs", STATE)
        if model is not None and model.fluid.vapoil:
            self.depends_on("rv", STATE)

    def evaluate_on_domain(self, model: Any, state: Any) -> List[List[Optional[Any]]]:
        return [c.get_component_mobility(model, state) for c in model.components]


__all__ = ["ComponentMobility", "ComponentPhaseDensity", "ComponentProperty"]


if __name__ == "__main__":
    import numpy as np

    from pymrst.gridprocessing import cart_grid, compute_geometry
    from pymrst.modules.ad_blackoil.models.generic_black_oil_model import (
        GenericBlackOilModelAD,
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
        model = GenericBlackOilModelAD(g, rock, fluid, w)
        st = model.init_state_ad(State(
            pressure=np.linspace(210, 190, g.cells.num) * BARSA,
            sW=np.full(g.cells.num, 0.45), qWs=np.array([2e-3]),
            qOs=np.array([-2e-3]), bhp=np.array([150 * BARSA])))
        return model, st

    def test_phase_density_matches_component_density():
        """ComponentPhaseDensity[c][ph] is the component's own density."""
        model, st = _ow_model()
        model.FlowPropertyFunctions.set_state_function(
            "ComponentPhaseDensity", ComponentPhaseDensity(model))
        cpd = model.get_props(st, "ComponentPhaseDensity")
        b = model.get_props(st, "ShrinkageFactors")
        for c, comp in enumerate(model.components):
            direct = comp.component_density(model, st, b, None, None)
            for ph in range(model.n_phases):
                if direct[ph] is None:
                    assert cpd[c][ph] is None
                else:
                    assert np.allclose(value(cpd[c][ph]), value(direct[ph]))
        print("ok test_phase_density_matches_component_density")

    def test_mobility_is_mob_times_density():
        """ComponentMobility[c][ph] = Mobility[ph] * ComponentPhaseDensity."""
        model, st = _ow_model()
        model.FlowPropertyFunctions.set_state_function(
            "ComponentMobility", ComponentMobility(model))
        model.FlowPropertyFunctions.set_state_function(
            "ComponentPhaseDensity", ComponentPhaseDensity(model))
        cmob = model.get_props(st, "ComponentMobility")
        cpd = model.get_props(st, "ComponentPhaseDensity")
        mob = model.get_props(st, "Mobility")
        for c in range(len(model.components)):
            for ph in range(model.n_phases):
                if cpd[c][ph] is None:
                    assert cmob[c][ph] is None
                else:
                    expect = mob[ph] * cpd[c][ph]
                    assert np.allclose(value(cmob[c][ph]), value(expect))
                    assert np.allclose((cmob[c][ph] - expect).jac[0].toarray(),
                                       0.0)
        print("ok test_mobility_is_mob_times_density")

    def test_component_mass_matches_graph_mass():
        """get_component_mass reproduces s*pv*density per phase."""
        model, st = _ow_model()
        s = model.get_props(st, "Saturations")
        pv = model.get_props(st, "PoreVolume")
        for comp in model.components:
            dens = comp.get_component_density(model, st)
            mass = comp.get_component_mass(model, st)
            for ph in range(model.n_phases):
                if dens[ph] is None:
                    assert mass[ph] is None
                else:
                    assert np.allclose(value(mass[ph]),
                                       value(s[ph] * pv * dens[ph]))
        print("ok test_component_mass_matches_graph_mass")

    test_phase_density_matches_component_density()
    test_mobility_is_mob_times_density()
    test_component_mass_matches_graph_mass()
    print("\nAll tests passed!")
