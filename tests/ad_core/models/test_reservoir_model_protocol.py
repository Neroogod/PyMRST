"""MRST's `ReservoirModel` protocol, ported (Phase 2).

The base carried 6 of MRST's 55 methods. What was missing splits into three
kinds, and the first is why this phase matters more than its size suggests:

* **saturation updates** - `updateSaturations` and `adjustStepFromSatBounds`
  are the rules that keep a Newton step inside the physical range. They pair
  with Phase 1's update limiting: that one bounds *how far* a variable may
  move, these bound *where it may end up*, and saturation needs both;
* **relative permeability** - pure functions of saturation, exactly checkable,
  including the three-phase oil interpolation that has no closed form of its
  own and is built from the two-phase curves;
* **indexing and storage** - which phase is which column, and writing computed
  values back onto the state.

The rest of `ReservoirModel` - well and boundary-condition insertion into the
equation system - is in `test_reservoir_model_forces.py`. It was deferred out
of this phase for needing an assembly protocol PyMRST did not have; the
protocol arrived with `LinearizedProblem.from_equations` and the facility
model's `get_well_contributions`, and the group was ported against those.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from pymrst.modules.ad_core.models.reservoir_model import ReservoirModel
from pymrst.modules.ad_core.statefunctions.state_function import State


class _Model(ReservoirModel):
    """A ReservoirModel with the phases named, and nothing else built.

    Subclassed rather than constructed because `ReservoirModel.__init__` builds
    three state-function groupings from a real fluid; none of the behaviour
    under test needs them.
    """

    def __init__(self, phases, n_cells=4):
        self.fluid = SimpleNamespace(phases=phases, n_phases=len(phases))
        self.G = SimpleNamespace(cells=SimpleNamespace(num=n_cells),
                                 faces=SimpleNamespace(num=n_cells + 1),
                                 griddim=3)
        self.operators = SimpleNamespace()
        self.gdz = None
        self.gravity_magnitude = 9.80665
        self.ds_max_abs = 0.2
        self.groupings = []
        self.state_function_evaluation_mode = "lazy"
        self.state_function_graph = None


# -- phase presence and indexing -------------------------------------------

def test_the_phase_triple_is_wog_shaped_whatever_the_active_count():
    """MRST indexes a fixed `[water, oil, gas]` triple.

    `get_active_phases` returns a mask of length `n_phases` instead, so on a
    two-phase model the two disagree - `[1 1 0]` against `[1 1]`. They coincide
    only at three phases, which is why the divergence went unnoticed. The
    triple is what every method here uses.
    """
    two = _Model(["water", "oil"])
    np.testing.assert_array_equal(two.get_phase_presence(), [True, True, False])
    assert len(two.get_active_phases()) == 2         # the pre-existing shape

    # And the triple must stay *computed*, not exposed as `model.water`. About
    # ten call sites probe `getattr(model, "water", False)`; since no concrete
    # model defined it, all of them had always received `False`. Exposing the
    # property flipped them to the real value - a behaviour change dressed as
    # an addition, and one this test now prevents from recurring.
    for name in ("water", "oil", "gas"):
        assert not hasattr(type(two), name), (
            f"exposing `model.{name}` flips every getattr(model, '{name}', "
            f"False) probe in the package")

    three = _Model(["water", "oil", "gas"])
    np.testing.assert_array_equal(three.get_phase_presence(), [True]*3)

    og = _Model(["oil", "gas"])
    np.testing.assert_array_equal(og.get_phase_presence(), [False, True, True])


def test_phase_indices_shift_down_when_a_phase_is_absent():
    """The cumulative-count rule. Dropping water moves oil and gas down one;
    a fixed table would put oil in water's column."""
    assert _Model(["water", "oil", "gas"]).get_phase_indices() == (1, 2, 3)
    assert _Model(["water", "oil"]).get_phase_indices() == (1, 2, -1)
    assert _Model(["oil", "gas"]).get_phase_indices() == (-1, 1, 2)
    assert _Model(["water", "gas"]).get_phase_indices() == (1, -1, 2)


def test_saturation_variable_names_follow_the_active_phases():
    assert _Model(["water", "oil"]).get_saturation_var_names() == ["sw", "so"]
    assert _Model(["oil", "gas"]).get_saturation_var_names() == ["so", "sg"]
    assert _Model(["water", "oil", "gas"]).get_saturation_var_names() == \
        ["sw", "so", "sg"]


def test_an_unknown_saturation_name_returns_minus_one_not_a_wrong_index():
    model = _Model(["water", "oil"])
    assert model.sat_var_index("so") == 1
    assert model.sat_var_index("SO") == 1
    assert model.sat_var_index("sg") == -1


def test_primary_variables_split_into_rest_saturations_and_wells():
    model = _Model(["water", "oil"])
    model.FacilityModel = SimpleNamespace(
        get_primary_variable_names=lambda: ["bhp", "qWs"])
    rest, sats, wells = model.split_primary_variables(
        ["pressure", "sw", "bhp", "qWs", "rs"])
    assert rest == ["pressure", "rs"]
    assert sats == ["sw"]
    assert wells == ["bhp", "qWs"]


# -- saturation bounds ------------------------------------------------------

def test_a_step_that_would_leave_the_unit_interval_is_shortened_not_clamped():
    """The distinction that matters.

    Clamping moves one phase and leaves the others, so the step stops pointing
    where the linear solve sent it. Scaling the whole row brings the worst
    offender exactly to its bound and keeps the direction.

    The increments deliberately do *not* sum to zero. With complementary
    increments (`+d, -d`) clamping and scaling give the identical answer, so a
    clamped implementation passes - confirmed by mutating this to a clamp and
    watching an earlier version of this test stay green.
    """
    s = np.array([[0.5, 0.5]])
    ds = np.array([[0.6, -0.1]])          # would put sw at 1.1
    adjusted = ReservoirModel.adjust_step_from_sat_bounds(s, ds)

    landing = s + adjusted
    assert landing.max() <= 1.0 + 1e-12
    assert landing.min() >= -1e-12
    # Direction preserved. A clamp would leave the second component at -0.1
    # and so change the ratio from -6 to -5.
    assert adjusted[0, 0]/adjusted[0, 1] == pytest.approx(ds[0, 0]/ds[0, 1])


def test_a_step_that_stays_in_bounds_is_untouched():
    s = np.array([[0.5, 0.5], [0.2, 0.8]])
    ds = np.array([[0.1, -0.1], [-0.05, 0.05]])
    np.testing.assert_allclose(
        ReservoirModel.adjust_step_from_sat_bounds(s, ds), ds)


def test_the_lower_bound_is_enforced_too():
    """Three phases, and only the lower bound violated.

    In a complementary two-phase pair, one phase going below zero means the
    other went above one, so an implementation that checked only the upper
    bound would still fix it - and pass. This case leaves every phase under
    one, so the lower bound is the only thing that can catch it.
    """
    s = np.array([[0.1, 0.5, 0.4]])
    ds = np.array([[-0.3, 0.1, 0.1]])      # sw to -0.2; nothing exceeds 1
    assert (s + ds).max() < 1.0, "this case must not violate the upper bound"

    adjusted = ReservoirModel.adjust_step_from_sat_bounds(s, ds)
    assert (s + adjusted).min() >= -1e-12


def test_only_the_offending_rows_are_scaled():
    """A shared factor would shorten every cell's step because one cell
    misbehaved, which is a global slowdown caused by one bad cell."""
    s = np.array([[0.5, 0.5], [0.99, 0.01]])
    ds = np.array([[0.1, -0.1], [0.5, -0.5]])
    adjusted = ReservoirModel.adjust_step_from_sat_bounds(s, ds)
    np.testing.assert_allclose(adjusted[0], ds[0])
    assert np.abs(adjusted[1]).max() < np.abs(ds[1]).max()


# -- relative permeability --------------------------------------------------

class _Fluid:
    """Analytic Corey curves, so every relperm result is known in closed form."""
    swcon = 0.0

    @staticmethod
    def kr_w(sw, **kw):
        return np.asarray(sw)**2

    @staticmethod
    def kr_g(sg, **kw):
        return np.asarray(sg)**2

    @staticmethod
    def kr_ow(so, **kw):
        return np.asarray(so)**3

    @staticmethod
    def kr_og(so, **kw):
        return np.asarray(so)**4


def test_two_phase_relperm_calls_the_right_curve_for_each_phase():
    sw, so = np.array([0.3]), np.array([0.7])
    kr_w, kr_o = ReservoirModel.rel_perm_wo(sw, so, _Fluid)
    np.testing.assert_allclose(kr_w, sw**2)
    np.testing.assert_allclose(kr_o, so**3)      # krOW, there being no krO

    kr_o, kr_g = ReservoirModel.rel_perm_og(so, sw, _Fluid)
    np.testing.assert_allclose(kr_o, so**4)      # krOG for the gas pair
    np.testing.assert_allclose(kr_g, sw**2)


def test_three_phase_oil_interpolates_between_the_two_phase_curves():
    """The one relperm with no measured curve of its own.

    Oil is interpolated between the oil-water and oil-gas curves by how much
    of the non-oil space each of water and gas occupies. Checked against the
    closed form rather than a golden number.
    """
    sw, so, sg = np.array([0.3]), np.array([0.5]), np.array([0.2])
    kr_w, kr_o, kr_g = ReservoirModel.rel_perm_wog(sw, so, sg, _Fluid)

    ww = (sw - 0.0)/(sg + sw - 0.0)
    expected = (1.0 - ww)*so**4 + ww*so**3
    np.testing.assert_allclose(kr_o, expected)
    np.testing.assert_allclose(kr_w, sw**2)
    np.testing.assert_allclose(kr_g, sg**2)


def test_the_three_phase_oil_curve_reduces_to_each_two_phase_limit():
    """A metamorphic check the interpolation cannot fake.

    With no gas the weight on water must be one, so the three-phase oil curve
    must equal krOW exactly; with no water above connate it must equal krOG.
    """
    so = np.array([0.5])
    _, kr_o, _ = ReservoirModel.rel_perm_wog(
        np.array([0.5]), so, np.array([0.0]), _Fluid)
    np.testing.assert_allclose(kr_o, so**3)

    # The gas limit is approached to O(1e-5), not exactly: MRST clips connate
    # water to `sw - 1e-5` to keep the denominator off zero, so a trace of the
    # water curve always remains. That is the formula's own resolution, not
    # slack in the test - at rtol 1e-6 this fails by 2e-5.
    _, kr_o, _ = ReservoirModel.rel_perm_wog(
        np.array([0.0]), so, np.array([0.5]), _Fluid)
    np.testing.assert_allclose(kr_o, so**4, rtol=1e-4)


def test_connate_water_is_clipped_below_the_saturation():
    """Otherwise a cell sitting at connate divides by zero."""
    class _Connate(_Fluid):
        swcon = 0.2

    # Gas must be absent as well: `sg + sw - swcon` is only zero when both the
    # gas saturation is zero *and* water sits exactly at connate. With any gas
    # present the denominator is finite and an unclipped implementation passes.
    sw, sg = np.array([0.2]), np.array([0.0])
    assert (sg + sw - _Connate.swcon).item() == 0.0, "the singular case"

    _, kr_o, _ = ReservoirModel.rel_perm_wog(sw, np.array([0.8]), sg, _Connate)
    assert np.all(np.isfinite(kr_o))


def test_evaluate_rel_perm_dispatches_on_the_active_phases():
    model = _Model(["water", "oil"])
    model.raw_fluid = _Fluid
    kr = model.evaluate_rel_perm([np.array([0.3]), np.array([0.7])])
    assert len(kr) == 2
    np.testing.assert_allclose(kr[0], 0.3**2)


def test_a_saturation_count_that_does_not_match_the_phases_is_refused():
    model = _Model(["water", "oil"])
    model.raw_fluid = _Fluid
    with pytest.raises(ValueError, match="active phases"):
        model.evaluate_rel_perm([np.array([0.3])])


# -- storing phase data -----------------------------------------------------

def test_phase_data_is_written_to_the_active_columns_only():
    """`data` is always the full WOG triple; a two-phase model must take the
    first and second and write two columns, not three."""
    model = _Model(["water", "oil"])
    state = State({"mob": np.zeros((4, 2))})
    model.store_mobilities(state, np.full(4, 1.0), np.full(4, 2.0),
                           np.full(4, 99.0))
    np.testing.assert_allclose(state["mob"][:, 0], 1.0)
    np.testing.assert_allclose(state["mob"][:, 1], 2.0)
    assert state["mob"].shape == (4, 2)


def test_a_gas_oil_model_skips_the_water_slot():
    """The trap: writing WOG data positionally would put water's values in
    oil's column for any model without water."""
    model = _Model(["oil", "gas"])
    state = State({})
    model.store_density(state, np.full(4, 111.0), np.full(4, 800.0),
                        np.full(4, 1.2))
    np.testing.assert_allclose(state["rho"][:, 0], 800.0)
    np.testing.assert_allclose(state["rho"][:, 1], 1.2)


def test_fluxes_are_stored_on_internal_connections_only():
    """MRST sizes `state.flux` by *every* connection and fills only the
    internal ones, so a boundary face keeps a zero rather than being dropped -
    the row index stays the face index, which is what plotting and export
    assume. An earlier version of this test expected the compacted shape.
    """
    model = _Model(["water", "oil"])
    model.operators = SimpleNamespace(
        internalConn=np.array([False, True, True, False, True]))
    state = State({})
    model.store_fluxes(state, np.full(3, 1.0), np.full(3, 2.0))

    assert state["flux"].shape == (5, 2)
    np.testing.assert_allclose(state["flux"][[1, 2, 4], 0], 1.0)
    np.testing.assert_allclose(state["flux"][[1, 2, 4], 1], 2.0)
    np.testing.assert_allclose(state["flux"][[0, 3], :], 0.0)


def test_upstream_flags_are_boolean_and_sized_by_the_interfaces():
    model = _Model(["water", "oil"])
    model.operators = SimpleNamespace(N=np.zeros((7, 2), dtype=int))
    state = State({})
    model.store_upstream_indices(state, np.ones(7), np.zeros(7))
    assert state["upstreamFlag"].shape == (7, 2)
    assert state["upstreamFlag"].dtype == bool


# -- gravity ----------------------------------------------------------------

def test_the_gravity_vector_is_truncated_to_the_grid_dimension():
    model = _Model(["water", "oil"])
    model.G.griddim = 2
    assert model.get_gravity_vector().size == 2


def test_the_gravity_gradient_prefers_the_operator_the_fluxes_use():
    """Recomputing from centroids would silently disagree with the operator
    the discretization actually applies."""
    model = _Model(["water", "oil"])
    model.operators = SimpleNamespace(gdz=np.array([1.0, 2.0, 3.0]))
    np.testing.assert_allclose(model.get_gravity_gradient(), [1.0, 2.0, 3.0])


# -- what is deliberately not here ------------------------------------------

def test_the_update_limits_carry_mrsts_defaults():
    """Only the saturation limit is finite. Saturation is bounded with
    tabulated properties; pressure is unbounded and smooth, so MRST leaves it
    free rather than guessing a scale."""
    limits = ReservoirModel.UPDATE_LIMITS
    assert limits["ds_max_abs"] == 0.2
    assert limits["dp_max_rel"] == float("inf")
    assert limits["dp_max_abs"] == float("inf")
    assert limits["minimum_pressure"] == float("-inf")
