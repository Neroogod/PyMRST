"""A control step may open or shut a well (MRST schedule semantics).

Field schedules routinely drill a producer partway through a run, or abandon
one. PyMRST refused it: the per-well state arrays are indexed positionally, so
a changed well count meant `qWs[i]` and `wells[i]` no longer described the same
well, and the solver raised rather than produce a silent shape error.

`_realign_well_state` rebuilds those arrays, and the load-bearing decision is
that wells are matched **by name**, not by position. Abandoning the second of
three leaves the third in slot two; carrying the arrays over positionally hands
the third well the second's rates and pressure. That is not a crash - it is a
plausible starting guess for the wrong well, which Newton converges away from,
so the run completes and the first step of that control period is quietly
wrong.

A surviving well keeps its converged solution, which is a far better start than
any default. A new well gets the field's mean pressure and zero rates, and is
presented to the injector seeding as a well that was previously *shut* - so an
injector opening at a high rate gets its rates seeded from its own composition
rather than starting from zero.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from pymrst.modules.ad_core.simulators.nonlinear_solver import NonLinearSolver


def _well(name, status=True, control="bhp", val=2.0e7, compi=(1.0, 0.0, 0.0)):
    return SimpleNamespace(name=name, status=status, type=control, val=val,
                           compi=np.asarray(compi, dtype=float))


def _state():
    return {"qWs": np.array([1.0, -2.0, -3.0]),
            "qOs": np.array([0.0, -4.0, -5.0]),
            "qGs": np.zeros(3),
            "bhp": np.array([3.0e7, 2.0e7, 1.9e7])}


THREE = [_well("I1"), _well("P1"), _well("P2")]


# -- the anchor --------------------------------------------------------------

def test_a_surviving_well_keeps_its_own_solution_not_its_neighbours():
    """The defect the whole change exists to avoid.

    Dropping `P1` leaves `P2` in slot 1. Positional carry-over would give it
    `P1`'s rate of -2; it must keep its own -3.
    """
    state, aligned = NonLinearSolver._realign_well_state(
        _state(), THREE, [_well("I1"), _well("P2")])

    np.testing.assert_allclose(state["qWs"], [1.0, -3.0])
    np.testing.assert_allclose(state["qOs"], [0.0, -5.0])
    np.testing.assert_allclose(state["bhp"], [3.0e7, 1.9e7])

    # The predecessors handed to the injector seeding must follow the same
    # matching. On an *added* well positional and by-name agree, so only a
    # removal can tell them apart - `P2`'s predecessor is `P2`, not `P1`.
    assert [w.name for w in aligned] == ["I1", "P2"]


def test_reordering_the_same_wells_follows_the_names():
    """A control may list the same wells in a different order. Nothing about
    the well changed, so nothing about its solution should."""
    state, _ = NonLinearSolver._realign_well_state(
        _state(), THREE, [_well("P2"), _well("I1"), _well("P1")])
    # Same count, so the arrays are returned untouched - the solver's own
    # positional bookkeeping still lines up with the facility's well list.
    np.testing.assert_allclose(state["qWs"], [1.0, -2.0, -3.0])


def test_an_added_well_starts_from_the_field_mean_pressure():
    """Zero would be a pressure no cell is at, and the first residual would be
    enormous. The mean of the existing wells is a defensible start."""
    state, _ = NonLinearSolver._realign_well_state(
        _state(), THREE, THREE + [_well("P3")])

    assert state["qWs"].size == 4
    np.testing.assert_allclose(state["qWs"][:3], [1.0, -2.0, -3.0])
    assert state["qWs"][3] == 0.0
    assert state["bhp"][3] == pytest.approx(np.mean([3.0e7, 2.0e7, 1.9e7]))


def test_an_added_well_is_presented_as_previously_shut():
    """So the injector seeding sees a status change and supplies its rates. A
    new high-rate injector starting from zero gives Newton a violent first
    residual for no reason."""
    _, aligned = NonLinearSolver._realign_well_state(
        _state(), THREE, THREE + [_well("P3")])

    assert len(aligned) == 4
    assert aligned[3].status is False
    assert [w.name for w in aligned[:3]] == ["I1", "P1", "P2"]


def test_a_new_injector_gets_its_rates_seeded():
    """End to end: realign, then seed. The two are ordered deliberately - the
    seeding indexes positionally and must see arrays that already match."""
    new = THREE + [_well("I2", control="rate", val=5.0,
                         compi=(0.0, 0.0, 1.0))]
    state, aligned = NonLinearSolver._realign_well_state(_state(), THREE, new)
    seeded = NonLinearSolver._seed_injector_control_state(state, aligned, new)

    assert seeded["qGs"][3] == pytest.approx(5.0)
    assert seeded["qWs"][3] == pytest.approx(0.0)


# -- the unchanged path ------------------------------------------------------

def test_an_unchanged_well_count_is_left_completely_alone():
    """The common case must stay bit-identical: a control that only changes
    targets touches no array."""
    original = _state()
    state, aligned = NonLinearSolver._realign_well_state(
        original, THREE, [_well("I1"), _well("P1"), _well("P2")])

    assert state is original, "the state object was rebuilt unnecessarily"
    assert aligned == THREE


def test_a_state_without_per_well_arrays_is_untouched():
    state = {"pressure": np.zeros(4)}
    out, _ = NonLinearSolver._realign_well_state(state, THREE, THREE[:2])
    assert out is state


# -- when names cannot identify a well ---------------------------------------

def test_unnamed_wells_fall_back_to_reinitialising_everything():
    """Without names there is no identity to match on. Guessing a
    correspondence would be the positional carry-over this exists to avoid, so
    every well is re-initialised instead - worse start, right well."""
    unnamed = [_well(None), _well(None), _well(None)]
    state, aligned = NonLinearSolver._realign_well_state(
        _state(), unnamed, unnamed[:2])

    np.testing.assert_allclose(state["qWs"], [0.0, 0.0])
    assert all(w.status is False for w in aligned)


def test_duplicate_names_are_treated_as_unmatchable():
    """Two wells called the same thing give an ambiguous match, and picking
    either one is a guess."""
    duplicated = [_well("P1"), _well("P1"), _well("I1")]
    state, _ = NonLinearSolver._realign_well_state(
        _state(), duplicated, [_well("P1"), _well("I1")])
    np.testing.assert_allclose(state["qWs"], [0.0, 0.0])


# -- the solver no longer refuses --------------------------------------------

def test_applying_a_control_with_a_different_well_count_is_accepted():
    """The refusal this replaces. It raised `NotImplementedError`, which made
    an ordinary field schedule unrunnable."""
    facility = SimpleNamespace(wells=list(THREE), n_wells=3,
                               within_limits=False)
    model = SimpleNamespace(facility=facility, bc=None, src=None)
    control = SimpleNamespace(W=[_well("I1"), _well("P2")], bc=None, src=None)

    NonLinearSolver._apply_control(model, control)
    assert [w.name for w in facility.wells] == ["I1", "P2"]
