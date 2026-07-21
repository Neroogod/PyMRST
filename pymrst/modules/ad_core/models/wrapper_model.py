"""WrapperModel (MRST ad-core/models).

Ports MRST's `WrapperModel.m`: a model that wraps a *parent* model and forwards
every call to it. On its own it is a transparent pass-through - the wrapped
model behaves exactly like the parent. Its purpose is to be subclassed: a
sequential pressure model, a transport model, or a multiscale model wraps a
fully-implicit reservoir model and overrides only the few methods that differ,
inheriting everything else unchanged.

This is the base the sequential and multiscale families are built on, so it is
part of the model foundation even though it adds no physics of its own. The
delegation is explicit (not `__getattr__` magic) for the methods a model driver
calls, and a catch-all forwards anything else to the parent.
"""

from __future__ import annotations

from typing import Any

from pymrst.modules.ad_core.models.physical_model import PhysicalModel
from pymrst.modules.ad_core.statefunctions.state_function import State


class WrapperModel(PhysicalModel):
    """A transparent wrapper around a parent model; subclass to specialise."""

    def __init__(self, parent_model: Any):
        super().__init__()
        self.parent_model = parent_model

    # -- the reservoir model underneath (MRST getReservoirModel) ------------
    def get_reservoir_model(self) -> Any:
        parent = self.parent_model
        return parent.get_reservoir_model() if hasattr(parent,
                                                       "get_reservoir_model") \
            else parent

    # -- explicit delegation of the model driver surface --------------------
    def conservation_equations(self, state0: State, state: State, dt: float):
        return self.parent_model.conservation_equations(state0, state, dt)

    def get_model_equations(self, state0: State, state: State, dt: float):
        fn = getattr(self.parent_model, "get_model_equations", None)
        if fn is not None:
            return fn(state0, state, dt)
        return self.parent_model.conservation_equations(state0, state, dt)

    def get_props(self, state: State, *names: str) -> Any:
        return self.parent_model.get_props(state, *names)

    def init_state_ad(self, state: State) -> State:
        return self.parent_model.init_state_ad(state)

    def get_equations(self, state0: State, state: State, dt: float, *a, **kw):
        return self.parent_model.get_equations(state0, state, dt, *a, **kw)

    # -- catch-all: any attribute not overridden comes from the parent ------
    def __getattr__(self, name: str) -> Any:
        # __getattr__ only fires for names not found normally, so it will not
        # shadow the explicit methods above or `parent_model` itself.
        return getattr(object.__getattribute__(self, "parent_model"), name)


__all__ = ["WrapperModel"]


if __name__ == "__main__":
    import numpy as np

    from pymrst.gridprocessing import cart_grid, compute_geometry
    from pymrst.modules.ad_blackoil.models.generic_black_oil_model import (
        GenericBlackOilModel,
    )
    from pymrst.modules.ad_props import init_simple_adi_fluid
    from pymrst.params import add_well, make_rock
    from pymrst.utils.adi import value
    from pymrst.utils.units import BARSA, CENTI, DARCY, DAY, MILLI, POISE

    def _parent_and_states():
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
        st0 = State(pressure=np.full(g.cells.num, 200 * BARSA),
                    sW=np.full(g.cells.num, 0.4),
                    qWs=np.array([2e-3]), qOs=np.array([-2e-3]),
                    bhp=np.array([150 * BARSA]))
        return model, model.init_state_ad(st0), model.init_state_ad(st0)

    def test_wrapper_delegates_props():
        """get_props through the wrapper equals the parent's."""
        parent, st, _ = _parent_and_states()
        wrap = WrapperModel(parent)
        for name in ("Mobility", "PhaseFlux", "ComponentTotalMass"):
            a = wrap.get_props(st, name)
            b = parent.get_props(st, name)
            for i in range(len(a)):
                assert np.allclose(value(a[i]), value(b[i]))
        print("ok test_wrapper_delegates_props")

    def test_wrapper_delegates_equations():
        """conservation_equations through the wrapper equal the parent's."""
        parent, st, st0 = _parent_and_states()
        wrap = WrapperModel(parent)
        a = wrap.conservation_equations(st0, st, 30.0 * DAY)
        b = parent.conservation_equations(st0, st, 30.0 * DAY)
        for i in range(len(a)):
            assert np.allclose(value(a[i]), value(b[i]))
            assert np.allclose((a[i] - b[i]).jac[0].toarray(), 0.0)
        print("ok test_wrapper_delegates_equations")

    def test_catchall_forwards_attributes():
        """An attribute not overridden (n_phases, operators) comes from parent."""
        parent, _, _ = _parent_and_states()
        wrap = WrapperModel(parent)
        assert wrap.n_phases == parent.n_phases
        assert wrap.operators is parent.operators
        assert wrap.get_reservoir_model() is parent
        print("ok test_catchall_forwards_attributes")

    def test_subclass_can_override_one_method():
        """A subclass overriding one method keeps the rest delegated - the
        point of the wrapper (sequential/transport models are built this way)."""
        parent, st, st0 = _parent_and_states()

        class _HalfEquations(WrapperModel):
            def conservation_equations(self, s0, s, dt):
                base = self.parent_model.conservation_equations(s0, s, dt)
                return [0.5 * e for e in base]

        wrap = _HalfEquations(parent)
        a = wrap.conservation_equations(st0, st, 30.0 * DAY)
        b = parent.conservation_equations(st0, st, 30.0 * DAY)
        assert np.allclose(value(a[0]), 0.5 * value(b[0]))
        # but props still delegate unchanged
        assert np.allclose(value(wrap.get_props(st, "Mobility")[0]),
                           value(parent.get_props(st, "Mobility")[0]))
        print("ok test_subclass_can_override_one_method")

    test_wrapper_delegates_props()
    test_wrapper_delegates_equations()
    test_catchall_forwards_attributes()
    test_subclass_can_override_one_method()
    print("\nAll tests passed!")
