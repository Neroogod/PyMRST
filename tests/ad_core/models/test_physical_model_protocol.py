"""MRST's `PhysicalModel` protocol, ported (Phase 1).

The base carried 4 of MRST's 49 methods. The AD-OO machinery centralised there
- property access by name, Newton update limiting, state-function container
lifecycle, schedule validation - was missing, so each concrete model that
needed a piece of it grew its own, and update limiting existed in exactly one
module in the whole package.

Everything here is new behaviour on a base class that existing models inherit,
which makes one test more important than the rest: `test_nothing_that_existed_
changed`. A base class is the easiest place in a codebase to change something
by accident.
"""

from __future__ import annotations

import numpy as np
import pytest

from pymrst.modules.ad_core.models.physical_model import PhysicalModel
from pymrst.modules.ad_core.statefunctions.state_function import State


class _Model(PhysicalModel):
    """A model with two stored variables, one of them two-column."""

    def get_variable_field(self, name, strict=True):
        mapping = {"pressure": ("pressure", slice(None)),
                   "sW": ("s", 0),
                   "sO": ("s", 1)}
        if name in mapping:
            return mapping[name]
        if strict:
            raise KeyError(name)
        return None, None


def _state():
    return State({"pressure": np.array([100.0, 200.0, 300.0]),
                  "s": np.array([[0.2, 0.8], [0.5, 0.5], [0.9, 0.1]])})


# -- the invariant ---------------------------------------------------------

def test_nothing_that_existed_changed():
    """The four methods that were here keep their exact behaviour.

    `update_state` in particular stays abstract. MRST's default is a loop over
    the primary variables calling `updateStateFromIncrement`, and adopting it
    would silently change what a subclass without an override does - that
    belongs to the migration phase, not to an additive one.
    """
    model = PhysicalModel()
    assert model.groupings == []

    with pytest.raises(NotImplementedError):
        model.get_equations(None, None, 1.0)
    with pytest.raises(NotImplementedError):
        model.update_state(None, None)
    with pytest.raises(NotImplementedError):
        model.get_convergence(None, 1.0)

    state = State({"pressure": np.arange(3.0)})
    np.testing.assert_array_equal(model.get_props(state, "pressure"),
                                  state["pressure"])
    with pytest.raises(KeyError):
        model.get_props(state, "nope")


# -- property access -------------------------------------------------------

def test_a_column_of_a_multi_column_field_is_addressed_by_name():
    model, state = _Model(), _state()
    np.testing.assert_allclose(model.get_prop(state, "sW"), [0.2, 0.5, 0.9])
    np.testing.assert_allclose(model.get_prop(state, "sO"), [0.8, 0.5, 0.1])


def test_writing_one_column_leaves_the_other_alone():
    """A set that overwrote the whole field would silently drop the other
    phase, and the saturations would still sum to something plausible."""
    model, state = _Model(), _state()
    model.set_prop(state, "sW", np.array([0.3, 0.3, 0.3]))
    np.testing.assert_allclose(state["s"][:, 0], [0.3, 0.3, 0.3])
    np.testing.assert_allclose(state["s"][:, 1], [0.8, 0.5, 0.1])


def test_a_stored_variable_is_not_shadowed_by_a_state_function():
    """MRST's order: the variable field first, groupings second."""
    model, state = _Model(), _state()
    np.testing.assert_allclose(model.get_prop(state, "pressure"),
                               [100.0, 200.0, 300.0])


def test_set_props_and_increment_prop_round_trip():
    model, state = _Model(), _state()
    model.set_props(state, ["pressure", "sW"],
                    [np.array([1.0, 2.0, 3.0]), np.array([0.4, 0.4, 0.4])])
    np.testing.assert_allclose(model.get_prop(state, "pressure"), [1, 2, 3])
    model.increment_prop(state, "pressure", 10.0)
    np.testing.assert_allclose(model.get_prop(state, "pressure"), [11, 12, 13])


def test_capping_bounds_a_property_on_one_or_both_sides():
    model, state = _Model(), _state()
    model.set_prop(state, "pressure", np.array([-5.0, 50.0, 500.0]))
    model.cap_property(state, "pressure", 0.0)
    np.testing.assert_allclose(model.get_prop(state, "pressure"), [0, 50, 500])
    model.cap_property(state, "pressure", 0.0, 100.0)
    np.testing.assert_allclose(model.get_prop(state, "pressure"), [0, 50, 100])


def test_a_wrong_shaped_property_is_refused_with_both_numbers():
    model, state = _Model(), _state()
    model.check_property(state, "pressure", 3, 1)
    with pytest.raises(ValueError, match="expected 7, state had 3"):
        model.check_property(state, "pressure", 7, 1)
    with pytest.raises(KeyError, match="missing"):
        model.check_property(state, "absent", 3, 1)


# -- Newton update limiting ------------------------------------------------

def test_a_relative_limit_scales_the_update_not_the_variable():
    """MRST's rule: the cap is on how far a value may move relative to itself."""
    dv = np.array([[100.0], [1.0]])
    val = np.array([[100.0], [100.0]])
    limited, change = PhysicalModel.limit_update_relative(dv, val, 0.1)
    np.testing.assert_allclose(limited.ravel(), [10.0, 1.0])
    np.testing.assert_allclose(change, [0.1, 1.0])


def test_a_row_is_scaled_by_one_factor_so_the_update_keeps_its_direction():
    """The part that is easy to get wrong.

    Scaling each column independently would shorten one component more than
    another - a rotated step, not a shorter one - and the Newton direction
    would no longer be the one the linear solve produced.
    """
    dv = np.array([[10.0, 5.0]])
    limited, change = PhysicalModel.limit_update_absolute(dv, 2.0)
    np.testing.assert_allclose(limited, [[2.0, 1.0]])
    assert limited[0, 0]/limited[0, 1] == pytest.approx(dv[0, 0]/dv[0, 1])


def test_an_update_within_the_limits_is_untouched():
    dv = np.array([[0.5], [0.25]])
    limited, change = PhysicalModel.limit_update_absolute(dv, 10.0)
    np.testing.assert_allclose(limited, dv)
    np.testing.assert_allclose(change, [1.0, 1.0])


def test_the_smaller_of_two_limits_wins():
    """Given both, the update must satisfy both - not whichever ran last."""
    model, state = _Model(), _state()
    model.set_prop(state, "pressure", np.array([100.0, 100.0, 100.0]))
    _, updated, previous = model.update_state_from_increment(
        state, np.array([50.0, 50.0, 50.0]), None, "pressure",
        rel_change_max=0.1,      # would allow +10
        abs_change_max=2.0)      # allows only +2
    np.testing.assert_allclose(updated, [102.0, 102.0, 102.0])
    np.testing.assert_allclose(previous, [100.0, 100.0, 100.0])


def test_an_unlimited_update_is_applied_whole():
    model, state = _Model(), _state()
    model.set_prop(state, "pressure", np.array([1.0, 2.0, 3.0]))
    _, updated, _ = model.update_state_from_increment(
        state, np.array([10.0, 10.0, 10.0]), None, "pressure")
    np.testing.assert_allclose(updated, [11.0, 12.0, 13.0])


def test_a_relative_limit_freezes_a_zero_valued_entry():
    """MRST's documented trap, reproduced rather than fixed.

    `dv/0` is infinite, so the scaling for that row is zero and the value can
    never leave zero. `PhysicalModel.m` warns about exactly this - relative
    limits "will, in particular, prevent zero states from being updated, so use
    with care" - so the faithful port keeps the behaviour and this test records
    it. An earlier version of this test asserted the opposite, on the
    assumption that freezing was a bug in the port; it is MRST's.

    Freezing and NaN are different failures and need different inputs, which
    an earlier version of this test missed by only covering the first. A
    non-zero increment on a zero value gives `1/0 = inf`, so the scaling is
    zero and the row freezes - unpleasant but finite. A *zero* increment on a
    zero value gives `0/0 = NaN`, and without the guard the NaN propagates
    through the whole row and into the state, which is far worse than freezing
    because it destroys values that were never being updated.
    """
    dv = np.array([[1.0], [1.0]])
    val = np.array([[0.0], [10.0]])
    limited, change = PhysicalModel.limit_update_relative(dv, val, 0.5)
    assert limited[0, 0] == 0.0, "MRST freezes a zero-valued entry"
    assert limited[1, 0] == pytest.approx(1.0), "the other row is under its limit"

    # The case the guard exists for: an untouched cell that is exactly zero.
    still = np.array([[0.0], [1.0]])
    zero_val = np.array([[0.0], [10.0]])
    limited, change = PhysicalModel.limit_update_relative(still, zero_val, 0.5)
    assert np.all(np.isfinite(limited)), "0/0 must not put NaN into the state"
    assert np.all(np.isfinite(change))


def test_an_increment_for_an_absent_variable_is_zero_not_an_error():
    """MRST's models ask for variables a given problem may not carry."""
    class _Problem:
        primary_variables = ["pressure", "sW"]

    model = _Model()
    assert model.get_increment([np.ones(3), np.zeros(3)], _Problem(),
                               "temperature") == 0.0
    np.testing.assert_allclose(
        model.get_increment([np.ones(3), np.zeros(3)], _Problem(), "sW"),
        np.zeros(3))


def test_strip_vars_removes_case_insensitively():
    kept, removed = PhysicalModel.strip_vars(["pressure", "sW", "bhp"],
                                             ["SW"])
    assert kept == ["pressure", "bhp"]
    assert removed == [False, True, False]


# -- state-function lifecycle ----------------------------------------------

def test_the_evaluation_mode_is_checked_when_it_is_set():
    model = PhysicalModel()
    assert model.get_state_function_evaluation_mode() == "lazy"
    model.set_state_function_evaluation_mode("full")
    assert model.get_state_function_evaluation_mode() == "full"
    with pytest.raises(ValueError, match="lazy"):
        model.set_state_function_evaluation_mode("eager")


def test_reduce_state_drops_caches_and_derivatives():
    """What survives between Newton iterations.

    A grouping cache left on a state whose primary variables have changed is a
    stale property that the next iteration will happily use.
    """
    from pymrst.utils.adi import init_variables_adi

    (p,) = init_variables_adi(np.array([1.0, 2.0, 3.0]))
    state = State({"pressure": p})
    state.containers["FlowProps"] = {"Mobility": np.ones(3)}

    model = PhysicalModel()
    reduced = model.reduce_state(state)
    assert reduced.containers == {}
    assert not hasattr(reduced["pressure"], "jac")
    np.testing.assert_allclose(reduced["pressure"], [1.0, 2.0, 3.0])

    kept = model.reduce_state(state, remove_containers=False)
    assert "FlowProps" in kept.containers


def test_validate_state_supplies_a_time():
    model = PhysicalModel()
    assert model.validate_state(State({}))["time"] == 0.0


# -- driving forces and schedule -------------------------------------------

def test_a_force_the_model_does_not_accept_is_refused():
    """Otherwise it is carried along, never read, and the run proceeds as
    though nothing had been asked for."""
    model = PhysicalModel()
    model.validate_driving_forces({"W": [], "bc": None})
    with pytest.raises(KeyError, match="polymer"):
        model.validate_driving_forces({"polymer": 1.0})


def test_a_schedule_is_checked_before_anything_runs():
    model = PhysicalModel()
    good = {"control": [{"W": []}, {"W": []}],
            "step": {"val": [1.0, 2.0, 3.0], "control": [1, 1, 2]}}
    assert model.validate_schedule(good) is good


@pytest.mark.parametrize("broken,message", [
    ({"step": {"val": [1.0], "control": [1]}}, "missing .control"),
    ({"control": [{}]}, "missing .step"),
    ({"control": [{}], "step": {"control": [1]}}, "missing .step.val"),
    ({"control": [{}], "step": {"val": [1.0]}}, "missing .step.control"),
    ({"control": [{}], "step": {"val": [1.0, 2.0], "control": [1]}}, "must match"),
    ({"control": [{}], "step": {"val": [1.0], "control": [4]}}, "largest control"),
    ({"control": [{}], "step": {"val": [-1.0], "control": [1]}}, "negative timesteps"),
    ({"control": [{}], "step": {"val": [1.0], "control": [0]}}, "one-based"),
])
def test_each_schedule_defect_is_named(broken, message):
    """Each fails at setup instead of on whichever step first hits it, which
    for a long schedule could be hours in."""
    with pytest.raises(ValueError, match=message):
        PhysicalModel().validate_schedule(broken)


# -- what is deliberately not finished -------------------------------------

def test_the_adjoint_protocol_says_what_it_needs():
    """Ported structurally; it needs `get_adjoint_equations`, which no model
    defines. Saying so beats a stub that returns something plausible."""
    with pytest.raises(NotImplementedError, match="get_adjoint_equations"):
        PhysicalModel().solve_adjoint(None, None, None, None, None, 1)


def test_a_model_without_primary_variables_says_so():
    with pytest.raises(NotImplementedError, match="primary variables"):
        PhysicalModel().get_primary_variables(State({}))


def test_the_mrst_spellings_resolve():
    """Code written against the MATLAB names has to find these."""
    for matlab, python in [("getProp", "get_prop"), ("setProp", "set_prop"),
                           ("capProperty", "cap_property"),
                           ("updateStateFromIncrement",
                            "update_state_from_increment"),
                           ("limitUpdateRelative", "limit_update_relative"),
                           ("validateSchedule", "validate_schedule"),
                           ("prepareReportstep", "prepare_report_step"),
                           ("reduceState", "reduce_state")]:
        assert getattr(PhysicalModel, matlab) is getattr(PhysicalModel, python)
