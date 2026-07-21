"""Flow-state builders (MRST ad-core/statefunctions/flux/flow_state).

Ports MRST's `FlowStateBuilder.m` (the fully-implicit default) and
`ImplicitFlowStateBuilder.m` (the same thing, named).

A *flow state* is the state the flux terms of a conservation law are evaluated
on. The default is the current (implicit) state: every mobility, density and
potential is taken at the new time level, and the whole system is solved
together - unconditionally stable, but every cell couples to its neighbours.

The builder exists so that choice can be changed *per property*: an explicit
builder freezes some flux terms at the old state (IMPES-style, cheaper per step
but CFL-limited), and an adaptive-implicit builder freezes them only in the
cells where the step is stable. The mechanism is the same in every case -
build a state whose property caches are pre-seeded with the chosen values, so
the graph returns them instead of recomputing from the implicit primaries.

This module is the base and the implicit builder; `ExplicitFlowStateBuilder`
and `AdaptiveImplicitFlowStateBuilder` are the two subclasses.
"""

from __future__ import annotations

from typing import Any, Optional

from pymrst.modules.ad_core.statefunctions.state_function import State


def owning_grouping(model: Any, prop: str):
    """The grouping on ``model`` that defines state function ``prop``."""
    for gname in model.groupings:
        grp = getattr(model, gname)
        if grp.has_state_function(prop):
            return grp
    raise KeyError(f"no grouping on the model defines '{prop}'")


def seed_property(flow_state: State, grouping: Any, prop: str, value: Any) -> None:
    """Pre-seed ``value`` for ``prop`` into a grouping's cache on the state.

    A seeded property is treated as already-evaluated: the graph returns it
    verbatim instead of recomputing it from the primary variables. This is the
    single mechanism every builder uses to make a term explicit.
    """
    flow_state.containers.setdefault(grouping.struct_name, {})[prop] = value


class FlowStateBuilder:
    """Fully-implicit flow state: every flux term at the new time level."""

    def __init__(self, verbose: bool = False):
        self.verbose = verbose

    def get_maximum_timestep(self, model: Any, state: State, state0: State,
                             dt: float, forces: Any = None) -> float:
        """Stability-limited maximum step. Implicit is unconditionally stable,
        so there is no limit."""
        return float("inf")

    def build(self, model: Any, state: State, state0: State, dt: float,
              fd: Any = None) -> State:
        """The flow state to evaluate fluxes on. Implicit: the state itself."""
        return state

    def prepare_timestep(self, model: Any, state: State, state0: State,
                         dt: float, forces: Any = None):
        """Hook run before each nonlinear loop. Nothing to do when implicit."""
        return self, state


class ImplicitFlowStateBuilder(FlowStateBuilder):
    """Explicitly the implicit builder.

    MRST keeps this as a named subclass so a model can ask for implicit flux
    states by class, even though the base already is implicit. Behaviour is
    identical to `FlowStateBuilder`.
    """


__all__ = [
    "FlowStateBuilder",
    "ImplicitFlowStateBuilder",
    "owning_grouping",
    "seed_property",
]


if __name__ == "__main__":
    import numpy as np

    from pymrst.gridprocessing import cart_grid, compute_geometry
    from pymrst.modules.ad_blackoil.models.generic_black_oil_model import (
        GenericBlackOilModel,
    )
    from pymrst.modules.ad_props import init_simple_adi_fluid
    from pymrst.params import add_well, make_rock
    from pymrst.utils.adi import value
    from pymrst.utils.units import BARSA, CENTI, DARCY, MILLI, POISE

    def _model_and_states():
        g = compute_geometry(cart_grid([6, 5, 2], [300.0, 250.0, 20.0]))
        rock = make_rock(g, 100 * MILLI * DARCY, 0.25)
        fluid = init_simple_adi_fluid(
            mu=np.array([1.0, 5.0]) * CENTI * POISE, rho=[1000.0, 800.0],
            n=[2, 2], c=[1e-6 / BARSA, 1e-4 / BARSA], p_ref=200 * BARSA)
        w = add_well(None, g, rock, np.array([0]), well_type="rate", val=2e-3,
                     compi=np.array([1.0, 0.0]), sign=1)
        w = add_well(w, g, rock, np.array([g.cells.num - 1]), well_type="bhp",
                     val=150 * BARSA, compi=np.array([0.0, 1.0]), sign=-1)
        model = GenericBlackOilModel(g, rock, fluid, w, gravity=9.80665)
        st0 = State(pressure=np.linspace(230, 180, g.cells.num) * BARSA,
                    sW=np.full(g.cells.num, 0.4),
                    qWs=np.array([2e-3]), qOs=np.array([-2e-3]),
                    bhp=np.array([150 * BARSA]))
        st = model.init_state_ad(st0)          # implicit AD state
        return model, st, st0

    def test_implicit_build_is_identity():
        """The implicit builder returns the state unchanged - the flux the
        graph computes on it is the fully-implicit flux."""
        model, st, st0 = _model_and_states()
        base = model.get_props(st, "PhaseFlux")
        fs = FlowStateBuilder().build(model, st, st0, dt=30.0)
        assert fs is st
        flux = model.get_props(fs, "PhaseFlux")
        for ph in range(model.n_phases):
            assert np.allclose(value(flux[ph]), value(base[ph]))
        print("ok test_implicit_build_is_identity")

    def test_implicit_has_no_timestep_limit():
        """Implicit is unconditionally stable: no CFL cap."""
        model, st, st0 = _model_and_states()
        dtm = ImplicitFlowStateBuilder().get_maximum_timestep(
            model, st, st0, dt=30.0)
        assert dtm == float("inf")
        print("ok test_implicit_has_no_timestep_limit")

    def test_seed_property_short_circuits_graph():
        """A seeded property is returned verbatim, not recomputed."""
        model, st, st0 = _model_and_states()
        fs = st.copy()
        grp = owning_grouping(model, "FaceMobility")
        sentinel = [np.full(model.operators.T.size, -123.0)
                    for _ in range(model.n_phases)]
        seed_property(fs, grp, "FaceMobility", sentinel)
        got = model.get_props(fs, "FaceMobility")
        assert all(np.allclose(value(got[i]), -123.0)
                   for i in range(model.n_phases))
        print("ok test_seed_property_short_circuits_graph")

    test_implicit_build_is_identity()
    test_implicit_has_no_timestep_limit()
    test_seed_property_short_circuits_graph()
    print("\nAll tests passed!")
