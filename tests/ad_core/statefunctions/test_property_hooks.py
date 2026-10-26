"""MRST's state-function region lookups and per-phase evaluation seams.

Four small groups that share one theme - the places where a subclass or a deck
is meant to change what a property computes:

* **region indicators** (`getRegionSaturation`, `getRegionSurfactant`,
  `getRegionPVT`). A deck with several regions has a different table set per
  region; reading the wrong one gives a plausible number from the wrong rock;
* **per-phase evaluation** (`evaluatePhaseViscosity`,
  `evaluatePhaseShrinkageFactor`). MRST loops the phases and subclasses
  override one phase. PyMRST evaluated all phases in one vectorised call, so
  there was no seam to override at all - the black-oil branches lived inside
  the fluid instead;
* **mobility multipliers** (`enableMultipliers`), whose real content is that
  enabling *declares a dependency*;
* **minimum derivatives** (`ensureMinimumDerivatives`), which keeps a Jacobian
  row from being structurally empty for an absent component.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import scipy.sparse as sp

from pymrst.gridprocessing import cart_grid, compute_geometry
from pymrst.modules.ad_blackoil.models.generic_black_oil_model import (
    GenericBlackOilModelAD,
)
from pymrst.modules.ad_core.statefunctions.flow_property_functions import (
    ComponentTotalMass, FlowPropertyFunctions, Mobility, region_indicator,
)
from pymrst.modules.ad_core.statefunctions.flowprops.shrinkage_factors import (
    BlackOilShrinkageFactors, ShrinkageFactors,
)
from pymrst.modules.ad_core.statefunctions.flowprops.viscosity import (
    BlackOilViscosity, Viscosity,
)
from pymrst.modules.ad_core.statefunctions.pvt_property_functions import (
    PVTPropertyFunctions,
)
from pymrst.modules.ad_core.statefunctions.state_function import State
from pymrst.modules.ad_props import init_simple_adi_fluid
from pymrst.params import add_well, make_rock
from pymrst.utils.adi import init_variables_adi, value
from pymrst.utils.units import BARSA, CENTI, DARCY, MILLI, POISE


def _model(nx=5, ny=4, nz=2):
    g = compute_geometry(cart_grid([nx, ny, nz], [250.0, 200.0, 20.0]))
    rock = make_rock(g, 100*MILLI*DARCY, 0.25)
    fluid = init_simple_adi_fluid(
        mu=np.array([1.0, 5.0])*CENTI*POISE, rho=[1000.0, 800.0],
        n=[2, 2], c=[1e-6/BARSA, 1e-4/BARSA], p_ref=200*BARSA)
    w = add_well(None, g, rock, np.array([0]), well_type="rate", val=2e-3,
                 compi=np.array([1.0, 0.0]), sign=1)
    w = add_well(w, g, rock, np.array([g.cells.num - 1]), well_type="bhp",
                 val=150*BARSA, compi=np.array([0.0, 1.0]), sign=-1)
    return GenericBlackOilModelAD(g, rock, fluid, w)


def _state(model):
    n = model.G.cells.num
    return model.init_state_ad(State(
        pressure=np.linspace(210, 190, n)*BARSA,
        sW=np.full(n, 0.45), qWs=np.array([2e-3]),
        qOs=np.array([-2e-3]), bhp=np.array([150*BARSA])))


# -- region indicators ------------------------------------------------------

def test_region_defaults_to_one_everywhere():
    """MRST returns `ones(nc, 1)` so a caller can always index the result."""
    model = _model()
    props = FlowPropertyFunctions(model)
    n = int(model.G.cells.num)

    np.testing.assert_array_equal(props.get_region_saturation(model),
                                  np.ones(n))
    np.testing.assert_array_equal(props.get_region_surfactant(model),
                                  np.ones(n))
    np.testing.assert_array_equal(
        PVTPropertyFunctions(model).get_region_pvt(model), np.ones(n))


@pytest.mark.parametrize("kind, getter", [
    ("saturation", "get_region_saturation"),
    ("surfactant", "get_region_surfactant"),
])
def test_a_declared_region_is_returned(kind, getter):
    model = _model()
    n = int(model.G.cells.num)
    declared = np.arange(n) % 3 + 1
    model.rock.regions = {kind: declared}

    np.testing.assert_array_equal(
        getattr(FlowPropertyFunctions(model), getter)(model), declared)


def test_the_pvt_region_is_read_from_its_own_key():
    """Each grouping reads *its* key. Sharing one would give the PVT tables
    the saturation regions, which is a wrong answer that still runs."""
    model = _model()
    model.rock.regions = {"saturation": np.full(model.G.cells.num, 2),
                          "pvt": np.full(model.G.cells.num, 3)}
    np.testing.assert_array_equal(
        PVTPropertyFunctions(model).get_region_pvt(model), 3)
    np.testing.assert_array_equal(
        FlowPropertyFunctions(model).get_region_saturation(model), 2)


def test_regions_are_read_under_either_spelling():
    """`rock.regions` may be a dict or an object; both occur.

    Two different keys, because a lookup that ignored its argument and always
    read `saturation` would pass a single-key test.
    """
    model = _model()
    sat = np.full(model.G.cells.num, 4)
    surf = np.full(model.G.cells.num, 9)
    model.rock.regions = SimpleNamespace(saturation=sat, surfactant=surf)
    np.testing.assert_array_equal(region_indicator(model, "saturation"), sat)
    np.testing.assert_array_equal(region_indicator(model, "surfactant"), surf)


def test_absent_regions_stay_none_for_the_constructor():
    """`region_indicator` returns None and `get_region_*` returns ones, and
    the difference is load-bearing: None means "do not dispatch on regions at
    all", which is a different code path from a one-region dispatch."""
    model = _model()
    assert region_indicator(model, "saturation") is None


# -- per-phase evaluation ---------------------------------------------------

def test_the_per_phase_hook_agrees_with_the_vectorised_call():
    """The seam must not change the numbers it replaces."""
    model, state = _model(), None
    state = _state(model)
    p = state["pressure"]

    for prop, fluid_fn in ((Viscosity(model), model.fluid.mu),
                           (ShrinkageFactors(model), model.fluid.b)):
        expected = fluid_fn(p)
        for i, name in enumerate(model.get_phase_names()):
            hook = (prop.evaluate_phase_viscosity
                    if isinstance(prop, Viscosity)
                    else prop.evaluate_phase_shrinkage_factor)
            np.testing.assert_allclose(
                np.asarray(value(hook(model, state, name, p))),
                np.asarray(value(expected[i])))


def test_evaluate_on_domain_goes_through_the_hook():
    """Otherwise the override is dead code and the port is a wrapper."""
    model, state = _model(), None
    state = _state(model)
    seen = []

    prop = Viscosity(model)
    original = prop.evaluate_phase_viscosity

    def spy(model_, state_, name, p):
        seen.append(name)
        return original(model_, state_, name, p)

    prop.evaluate_phase_viscosity = spy
    prop.evaluate_on_domain(model, state)
    assert seen == list(model.get_phase_names())


def test_the_black_oil_hook_passes_the_dissolution_arguments():
    """The live-oil branch. Without `rs` the oil viscosity is the dead-oil
    curve - a plausible number, and wrong above the bubble point."""
    model = _model()
    state = _state(model)
    p = state["pressure"]

    plain = Viscosity(model).evaluate_phase_viscosity(model, state, "O", p)
    black = BlackOilViscosity(model).evaluate_phase_viscosity(
        model, state, "O", p)
    # This fluid is not live, so the two must agree exactly - which is what
    # makes the argument-passing safe to add to an immiscible model.
    np.testing.assert_allclose(np.asarray(value(black)),
                               np.asarray(value(plain)))


def test_a_subclass_can_override_a_single_phase():
    """The point of the seam: change one phase without touching the loop."""
    model = _model()
    state = _state(model)

    class OnlyWater(Viscosity):
        def evaluate_phase_viscosity(self, model_, state_, name, p):
            if name == "W":
                return np.full(int(model_.G.cells.num), 42.0)
            return super().evaluate_phase_viscosity(model_, state_, name, p)

    out = OnlyWater(model).evaluate_on_domain(model, state)
    np.testing.assert_allclose(np.asarray(value(out[0])), 42.0)
    np.testing.assert_allclose(
        np.asarray(value(out[1])),
        np.asarray(value(Viscosity(model).evaluate_on_domain(model, state)[1])))


def test_mrsts_own_curve_spelling_is_preferred():
    """A fluid carrying `muW`/`muO` is used the way its author intended."""
    model = _model()
    state = _state(model)
    model.raw_fluid = SimpleNamespace(muW=lambda p: np.full(np.size(value(p)), 7.0))

    got = Viscosity(model).evaluate_phase_viscosity(model, state, "W",
                                                    state["pressure"])
    np.testing.assert_allclose(np.asarray(value(got)), 7.0)


def test_the_black_oil_hook_reaches_the_live_oil_curve():
    """The immiscible fluid above cannot show this: its `b` ignores `rs`, so
    passing the dissolution arguments or dropping them gives the same number.

    A fluid that *does* branch on `rs` separates them, and this is the defect
    that matters - dropping `rs` silently returns the dead-oil curve, which is
    correct below the bubble point and wrong above it.
    """
    class _LiveFluid:
        disgas, vapoil, has_pc = True, False, False
        n_phases = 3
        phases = ["water", "oil", "gas"]
        rho_s = np.array([1000.0, 800.0, 1.2])

        def b(self, p, rs=None, sat=None, rv=None, sat_g=None):
            p = np.asarray(value(p), dtype=float)
            dead = np.ones_like(p)
            live = dead if rs is None else dead + np.asarray(value(rs), float)
            return [dead, live, dead]

    n = 4
    model = SimpleNamespace(
        fluid=_LiveFluid(), raw_fluid=None,
        G=SimpleNamespace(cells=SimpleNamespace(num=n)),
        n_phases=3,
        get_phase_names=lambda: ["W", "O", "G"],
        get_phase_index=lambda name: ["W", "O", "G"].index(name),
    )
    state = {"pressure": np.full(n, 2.0e7), "rs": np.full(n, 5.0),
             "status": np.zeros(n, dtype=int), "s": [np.full(n, 1/3)]*3}

    prop = BlackOilShrinkageFactors.__new__(BlackOilShrinkageFactors)
    prop.use_phase_pressures = False
    prop.disgas, prop.vapoil = True, False

    got = prop.evaluate_phase_shrinkage_factor(model, state, "O",
                                               state["pressure"])
    np.testing.assert_allclose(np.asarray(value(got)), 6.0), \
        "the dissolution arguments were not passed to the fluid"


def test_shrinkage_factors_use_the_same_seam():
    model = _model()
    state = _state(model)
    prop = BlackOilShrinkageFactors(model)
    out = prop.evaluate_on_domain(model, state)
    expected = model.fluid.b(state["pressure"])
    for i in range(model.n_phases):
        np.testing.assert_allclose(np.asarray(value(out[i])),
                                   np.asarray(value(expected[i])))


# -- mobility multipliers ---------------------------------------------------

def test_multipliers_are_off_by_default():
    assert Mobility(_model()).has_multipliers is False


def test_enabling_multipliers_declares_the_dependency():
    """The real content. The grouping evaluates in dependency order, so a
    multiplier used without being declared is read before it is computed -
    giving the previous iteration's value rather than an error."""
    mobility = Mobility(_model()).enable_multipliers()
    assert mobility.has_multipliers
    names = [d if isinstance(d, str) else getattr(d, "name", str(d))
             for d in mobility.dependencies]
    assert any("MobilityMultipliers" in n for n in names), names


def test_enabling_multipliers_returns_self_for_chaining():
    mobility = Mobility(_model())
    assert mobility.enable_multipliers() is mobility


# -- minimum derivatives ----------------------------------------------------

def _ad_mass(values, derivative):
    x, = init_variables_adi(np.asarray(values, dtype=float))
    n = x.val.size
    x.jac[0] = sp.csr_matrix(np.diag(np.full(n, derivative)))
    return x


def test_the_default_is_to_leave_derivatives_alone():
    """`None` is not a floor of zero: zero would still walk the Jacobian and
    rewrite entries that compare equal to it."""
    prop = ComponentTotalMass()
    assert prop.get_minimum_derivatives() is None
    mass = [_ad_mass([1.0, 2.0], 0.0)]
    out = prop.ensure_minimum_derivatives(mass)
    np.testing.assert_array_equal(out[0].jac[0].diagonal(), 0.0)


def test_a_floor_is_applied_to_small_derivatives():
    prop = ComponentTotalMass().set_minimum_derivatives(1e-4)
    mass = [_ad_mass([1.0, 2.0], 1e-9)]
    out = prop.ensure_minimum_derivatives(mass)
    np.testing.assert_allclose(out[0].jac[0].diagonal(), 1e-4)


def test_the_floor_preserves_the_sign():
    """A small negative derivative must not be flipped positive - that would
    reverse the direction the solver moves that cell."""
    prop = ComponentTotalMass().set_minimum_derivatives(1e-4)
    mass = [_ad_mass([1.0, 2.0], -1e-9)]
    out = prop.ensure_minimum_derivatives(mass)
    np.testing.assert_allclose(out[0].jac[0].diagonal(), -1e-4)


def test_large_derivatives_are_untouched():
    prop = ComponentTotalMass().set_minimum_derivatives(1e-10)
    mass = [_ad_mass([1.0, 2.0], 0.5)]
    out = prop.ensure_minimum_derivatives(mass)
    np.testing.assert_allclose(out[0].jac[0].diagonal(), 0.5)


def test_a_large_negative_derivative_is_untouched():
    """The comparison is on magnitude, not sign. A signed `d < floor` test
    would see -0.5 as "below 1e-4" and crush a perfectly good derivative to
    the floor - reversing nothing, but destroying the scale of that row."""
    prop = ComponentTotalMass().set_minimum_derivatives(1e-4)
    mass = [_ad_mass([1.0, 2.0], -0.5)]
    out = prop.ensure_minimum_derivatives(mass)
    np.testing.assert_allclose(out[0].jac[0].diagonal(), -0.5)


def test_the_default_floor_is_mrsts():
    assert ComponentTotalMass().set_minimum_derivatives().get_minimum_derivatives() \
        == pytest.approx(1e-10)


def test_a_per_component_floor_is_accepted():
    prop = ComponentTotalMass().set_minimum_derivatives([1e-4, 1e-2])
    mass = [_ad_mass([1.0], 1e-9), _ad_mass([1.0], 1e-9)]
    out = prop.ensure_minimum_derivatives(mass)
    np.testing.assert_allclose(out[0].jac[0].diagonal(), 1e-4)
    np.testing.assert_allclose(out[1].jac[0].diagonal(), 1e-2)


def test_a_plain_numeric_component_is_skipped():
    """A component with no AD has no derivative to floor."""
    prop = ComponentTotalMass().set_minimum_derivatives(1e-4)
    out = prop.ensure_minimum_derivatives([np.array([1.0, 2.0])])
    np.testing.assert_array_equal(out[0], [1.0, 2.0])
