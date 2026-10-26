"""MRST's `StateFunction` / `StateFunctionGrouping` protocol (Phase 2).

The graph machinery worked; what was missing was everything around it -
container lifecycle, region dispatch, and the output contract.

The output contract is the part that earns its place. A relative permeability
that returns 1.4, or a mobility that returns NaN, is wrong *at the moment it is
produced*, but nothing looks: the value goes into the residual, the linear
solve degrades, and the run fails several iterations later with the offending
property no longer on the state. `validate_output` names the property at the
point of failure. It warns rather than raises, deliberately - a property can be
transiently out of range mid-Newton and still converge - so the value is in the
message, not in stopping the run.

Validation is off by default. Every test that wants it turns it on, which is
also the proof that the default costs existing behaviour nothing.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest

from pymrst.modules.ad_core.statefunctions.state_function import (
    State, StateFunction, StateFunctionGrouping,
)


class _Constant(StateFunction):
    def __init__(self, values, label="Constant"):
        super().__init__(label=label)
        self.values = values

    def evaluate_on_domain(self, model, state):
        return self.values


class _Saturation(_Constant):
    """A property that declares a physical range, as a relperm would."""
    output_range = (0.0, 1.0)


def _grouping(**functions):
    group = StateFunctionGrouping("Props")
    for name, fn in functions.items():
        group.set_state_function(name, fn)
    return group


# -- the default costs nothing ---------------------------------------------

def test_validation_is_off_by_default():
    """Warning on every evaluation is right while debugging a property and
    wrong for a production run, so it is opt-in - which is also what makes
    this addition free for everything that existed before."""
    group = _grouping(bad=_Saturation(np.array([5.0])))
    assert group.validation_level == 0

    with warnings.catch_warnings():
        warnings.simplefilter("error")          # any warning fails the test
        group.evaluate_state_function(None, State({}), "bad")


# -- the output contract ----------------------------------------------------

def test_a_property_outside_its_declared_range_is_named():
    group = _grouping(kr=_Saturation(np.array([0.2, 1.4, 0.5])))
    group.set_validation_level(1)

    with pytest.warns(UserWarning, match="larger than 1"):
        group.evaluate_state_function(None, State({}), "kr")


def test_nan_and_inf_are_caught_even_with_no_range_declared():
    """The common failure: a property that divides by a zero mobility. No
    range is needed to know NaN is wrong."""
    group = _grouping(mob=_Constant(np.array([1.0, np.nan, np.inf])))
    group.set_validation_level(1)

    with pytest.warns(UserWarning) as caught:
        group.evaluate_state_function(None, State({}), "mob")
    messages = " ".join(str(w.message) for w in caught)
    assert "NaN" in messages and "Inf" in messages


def test_the_warning_names_the_property_and_counts_the_offenders():
    """"The solve diverged" against "Saturation produced 2 of 4 values above
    1" - the second one tells you where to look."""
    group = _grouping(s=_Saturation(np.array([0.1, 1.5, 0.3, 2.0])))
    group.set_validation_level(1)

    with pytest.warns(UserWarning) as caught:
        group.evaluate_state_function(None, State({}), "s")
    message = str(caught[0].message)
    assert "_Saturation" in message
    assert "2 of 4" in message


def test_a_property_inside_its_range_warns_about_nothing():
    group = _grouping(s=_Saturation(np.array([0.0, 0.5, 1.0])))
    group.set_validation_level(1)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        group.evaluate_state_function(None, State({}), "s")


def test_a_property_may_declare_that_nan_is_acceptable():
    class _Sparse(_Constant):
        allow_nan = True

    group = _grouping(x=_Sparse(np.array([1.0, np.nan])))
    group.set_validation_level(1)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        group.evaluate_state_function(None, State({}), "x")


# -- container lifecycle ----------------------------------------------------

def test_a_fresh_container_distinguishes_absent_from_unevaluated():
    """Collapsing the two makes a typo in a dependency name look like an
    ordering problem, which is a much harder thing to find."""
    group = _grouping(a=_Constant(1.0), b=_Constant(2.0))
    state = group.init_state_function_container(State({}))

    container = state.containers["Props"]
    assert set(container) == {"a", "b"}       # membership is known
    assert container["a"] is None             # evaluation is not
    assert "c" not in container               # and 'c' is simply not a member


def test_is_state_initialized_reports_the_container_not_the_values():
    group = _grouping(a=_Constant(1.0))
    state = State({})
    assert not group.is_state_initialized(state)
    group.init_state_function_container(state)
    assert group.is_state_initialized(state)


def test_evaluation_status_is_tracked_per_function():
    group = _grouping(a=_Constant(1.0), b=_Constant(2.0))
    state = State({})
    assert not group.is_state_function_evaluated(None, state, "a")

    group.evaluate_state_function(None, state, "a")
    assert group.is_state_function_evaluated(None, state, "a")
    assert not group.is_state_function_evaluated(None, state, "b")


def test_a_pre_evaluated_state_reports_everything_ready():
    """`evaluate_all_state_functions` marks the state, and the whole graph is
    then ready by construction."""
    group = _grouping(a=_Constant(1.0))
    state = State({"evaluated": True})
    assert group.is_state_function_evaluated(None, state, "anything")


# -- dependency checking ----------------------------------------------------

def test_a_dependency_on_a_function_that_does_not_exist_is_named():
    group = _grouping(a=_Constant(1.0).depends_on("missing"))
    with pytest.raises(KeyError, match="missing"):
        group.check_dependencies()


def test_an_external_dependency_is_checked_against_the_supplied_groupings():
    other = StateFunctionGrouping("Other")
    other.set_state_function("present", _Constant(1.0))
    group = _grouping(a=_Constant(1.0).depends_on("present", grouping="Other"))

    assert group.check_dependencies([other])

    other.remove_state_function("present")
    with pytest.raises(KeyError, match="Did not find present in group of class Other"):
        group.check_dependencies([other])
    assert group.check_dependencies([other], nargout=1) is False


def test_external_dependencies_are_recorded_separately():
    """`dependencies` keeps every edge so the graph walk is unchanged;
    `externals` is what resolves a bare name to its owning grouping."""
    fn = _Constant(1.0).depends_on("p", grouping="state") \
                       .depends_on("mu", grouping="PVTProps") \
                       .depends_on("local")
    assert len(fn.dependencies) == 3
    assert [d.name for d in fn.externals] == ["mu"]


def test_resetting_dependencies_clears_both_lists():
    fn = _Constant(1.0).depends_on("a").depends_on("b", grouping="Other")
    fn.reset_dependencies()
    assert fn.dependencies == [] and fn.externals == []


# -- region dispatch --------------------------------------------------------

def test_a_per_region_function_list_uses_each_cell_s_own_table():
    """Applying region one everywhere gives a plausible, wrong answer.

    The regions here return distinguishable values so a single-table
    implementation cannot produce this result by accident.
    """
    fn = _Constant(None)
    fn.set_regions(np.array([1, 1, 2, 2]))
    curves = [lambda x: x*10.0, lambda x: x*100.0]

    out = fn.evaluate_function_cell_subset(curves, None, np.array([1.0]*4))
    np.testing.assert_allclose(out, [10.0, 10.0, 100.0, 100.0])


def test_a_single_callable_is_applied_whole():
    fn = _Constant(None)
    out = fn.evaluate_function_cell_subset(lambda x: x + 1.0, None,
                                           np.array([1.0, 2.0]))
    np.testing.assert_allclose(out, [2.0, 3.0])


def test_a_region_list_without_a_region_map_is_refused():
    """Guessing which table to use is the one thing that must not happen."""
    fn = _Constant(None)
    with pytest.raises(ValueError, match="no .regions"):
        fn.evaluate_function_cell_subset([lambda x: x], None, np.array([1.0]))


def test_a_single_region_is_selected_by_one_based_index():
    fn = _Constant(None)
    curves = [lambda x: x*10.0, lambda x: x*100.0]
    assert fn.evaluate_function_single_region_with_arguments(curves, 2, 1.0) \
        == 100.0
    with pytest.raises(IndexError, match="outside the 2 regions"):
        fn.evaluate_function_single_region_with_arguments(curves, 3, 1.0)


def test_subsetting_a_grouping_restricts_every_member_s_regions():
    a = _Constant(1.0)
    a.set_regions(np.array([1, 2, 3, 4]))
    group = _grouping(a=a)
    sub = group.subset(np.array([0, 2]))
    np.testing.assert_array_equal(sub.get_state_function("a").regions, [1, 3])
    # MATLAB value semantics: the grouping subset is a copy, so a second
    # subset (the next equilibration region) starts from all the cells.
    np.testing.assert_array_equal(a.regions, [1, 2, 3, 4])
    np.testing.assert_array_equal(
        group.subset(np.array([3])).get_state_function("a").regions, [4])


# -- membership -------------------------------------------------------------

def test_restricting_a_grouping_to_named_functions_refuses_unknown_names():
    group = _grouping(a=_Constant(1.0), b=_Constant(2.0), c=_Constant(3.0))
    group.set_internal_names(["a", "c"])
    assert set(group.get_names()) == {"a", "c"}

    with pytest.raises(KeyError, match="nope"):
        group.set_internal_names(["nope"])


def test_the_mrst_spellings_resolve():
    for matlab, python in [
            ("getStateFunctionContainerName", "get_state_function_container_name"),
            ("initStateFunctionContainer", "init_state_function_container"),
            ("evaluateStateFunctionUnsafe", "evaluate_state_function_unsafe"),
            ("checkDependencies", "check_dependencies"),
            ("setValidationLevel", "set_validation_level"),
            ("evaluateStateFunctionWithDependencies", "evaluate_with_dependencies")]:
        assert getattr(StateFunctionGrouping, matlab) is \
            getattr(StateFunctionGrouping, python)
    for matlab, python in [("validateOutput", "validate_output"),
                           ("evaluateFluid", "evaluate_fluid"),
                           ("resetDependencies", "reset_dependencies")]:
        assert getattr(StateFunction, matlab) is getattr(StateFunction, python)
