"""MRST `NonLinearSolver` relaxation and convergence diagnosis (Phase 2).

PyMRST's solver already had a line search and a relaxation heuristic. What it
did not have is MRST's way of deciding *when* damping is warranted, which is a
different question from whether the residual went up:

* **oscillation** - the residual reversed direction. A step that overshoots
  produces this, and it is not divergence: a residual can oscillate while
  trending down. The right response is damping, not a timestep cut.
* **stagnation** - the residual stopped moving. Distinct from slow, and the
  equation that will still be there in ten more iterations.

Both are per-equation, because in a coupled system one residual stalling while
the others converge is the diagnosis - a single scalar norm hides it.

MRST's relaxation is also a different scheme from PyMRST's, not a rename: a
fixed parameter under a named type rather than a heuristic that adapts. Both
now exist, and the MRST path is inert at its defaults.
"""

from __future__ import annotations

import numpy as np
import pytest

from pymrst.modules.ad_core.simulators.nonlinear_solver import NonLinearSolver


def _solver(**kwargs):
    solver = NonLinearSolver(verbose=False)
    for key, item in kwargs.items():
        setattr(solver, key, item)
    return solver


# -- the defaults change nothing -------------------------------------------

def test_the_mrst_relaxation_path_is_inert_at_its_defaults():
    """A relaxation parameter of 1 must return the increment untouched, or
    every existing run changes."""
    solver = _solver()
    assert solver.relaxation_parameter == 1.0
    assert solver.relaxation_type == "dampen"

    dx = [np.array([1.0, 2.0]), np.array([3.0])]
    out, report = solver.stabilize_newton_increments(None, None, dx)
    np.testing.assert_array_equal(out[0], [1.0, 2.0])
    np.testing.assert_array_equal(out[1], [3.0])
    assert report["relaxation_parameter"] == 1.0


# -- oscillation ------------------------------------------------------------

def test_a_residual_that_reverses_direction_is_reported_as_oscillating():
    """Down then up. The ratio of successive changes goes negative."""
    res = np.array([[1.0], [0.1], [0.5]])
    assert _solver().check_for_oscillations(res, 2)[0]


def test_a_residual_descending_monotonically_is_not_oscillating():
    res = np.array([[1.0], [0.1], [0.01]])
    assert not _solver().check_for_oscillations(res, 2)[0]


def test_a_residual_rising_monotonically_is_not_oscillating_either():
    """Diverging is a different failure from oscillating, and wants a
    different response - a cut rather than damping."""
    res = np.array([[0.01], [0.1], [1.0]])
    assert not _solver().check_for_oscillations(res, 2)[0]


def test_oscillation_needs_three_iterations_before_it_can_say_anything():
    solver = _solver()
    res = np.array([[1.0], [0.5]])
    assert not solver.check_for_oscillations(res, 0).any()
    assert not solver.check_for_oscillations(res, 1).any()


def test_each_equation_is_judged_on_its_own():
    """A single scalar norm would hide one equation oscillating while the
    others descend, which is exactly the case worth knowing about."""
    res = np.array([[1.0, 1.0],
                    [0.1, 0.5],
                    [0.5, 0.2]])          # first oscillates, second descends
    flags = _solver().check_for_oscillations(res, 2)
    assert flags[0] and not flags[1]


# -- stagnation -------------------------------------------------------------

def test_a_residual_that_stops_moving_is_reported_as_stagnated():
    res = np.array([[1.0], [1.0 - 1e-6]])
    assert _solver(stagnate_tol=1e-2).check_for_stagnation(res, 1)[0]


def test_a_residual_still_falling_is_not_stagnated():
    res = np.array([[1.0], [0.1]])
    assert not _solver(stagnate_tol=1e-2).check_for_stagnation(res, 1)[0]


def test_stagnation_is_measured_relative_so_it_is_scale_free():
    """The same fractional change must read the same at any magnitude - an
    absolute test would call every small residual stagnated."""
    solver = _solver(stagnate_tol=1e-2)
    small = np.array([[1.0e-8], [1.0e-8*0.5]])
    large = np.array([[1.0e8], [1.0e8*0.5]])
    assert not solver.check_for_stagnation(small, 1)[0]
    assert not solver.check_for_stagnation(large, 1)[0]


def test_stagnation_needs_two_iterations():
    assert not _solver().check_for_stagnation(np.array([[1.0]]), 0).any()


# -- relaxation types -------------------------------------------------------

def test_dampen_scales_the_step_and_keeps_scaling_it():
    """Called twice on purpose.

    On the first call `previous_increment` is None, so a dampen branch that
    wrongly mixed with the previous step would fall through to plain scaling
    and give the right answer anyway. Only the second call distinguishes
    dampening from SOR - confirmed by mutating dampen into a mix and watching
    a single-call version of this test stay green.
    """
    solver = _solver(relaxation_type="dampen", relaxation_parameter=0.5)
    out, _ = solver.stabilize_newton_increments(None, None,
                                                 [np.array([2.0, 4.0])])
    np.testing.assert_allclose(out[0], [1.0, 2.0])

    out, _ = solver.stabilize_newton_increments(None, None,
                                                 [np.array([2.0, 4.0])])
    np.testing.assert_allclose(out[0], [1.0, 2.0],
                               err_msg="dampen must not mix with the previous step")


def test_sor_mixes_with_the_previous_step():
    """Damping shortens a step that is too long; SOR smooths a direction that
    is swinging between iterations. They are different remedies."""
    solver = _solver(relaxation_type="sor", relaxation_parameter=0.5)
    solver.stabilize_newton_increments(None, None, [np.array([10.0])])
    out, _ = solver.stabilize_newton_increments(None, None, [np.array([0.0])])
    np.testing.assert_allclose(out[0], [5.0])       # 0.5*0 + 0.5*10


def test_sor_falls_back_to_the_plain_step_on_the_first_iteration():
    """There is nothing to mix with yet."""
    solver = _solver(relaxation_type="sor", relaxation_parameter=0.5)
    out, _ = solver.stabilize_newton_increments(None, None, [np.array([4.0])])
    np.testing.assert_allclose(out[0], [4.0])


def test_none_leaves_the_step_alone_even_when_damped():
    """Also called twice, for the same reason: with no previous increment the
    SOR branch is skipped regardless, so one call cannot tell 'none' from a
    scheme that would have mixed."""
    solver = _solver(relaxation_type="none", relaxation_parameter=0.1)
    out, _ = solver.stabilize_newton_increments(None, None, [np.array([7.0])])
    np.testing.assert_allclose(out[0], [7.0])

    out, _ = solver.stabilize_newton_increments(None, None, [np.array([1.0])])
    np.testing.assert_allclose(out[0], [1.0])


def test_an_unknown_relaxation_type_is_refused_by_name():
    solver = _solver(relaxation_type="magic", relaxation_parameter=0.5)
    with pytest.raises(ValueError, match="magic"):
        solver.stabilize_newton_increments(None, None, [np.array([1.0])])


def test_the_previous_increment_is_remembered_whatever_the_type():
    """So switching to SOR mid-run has something to mix with."""
    solver = _solver(relaxation_type="dampen", relaxation_parameter=1.0)
    solver.stabilize_newton_increments(None, None, [np.array([3.0])])
    np.testing.assert_allclose(solver.previous_increment[0], [3.0])


# -- reporting --------------------------------------------------------------

def test_a_named_solver_prefixes_its_messages():
    assert _solver().get_id() == ""
    assert _solver(identifier="outer").get_id() == "outer: "


def test_the_line_search_judges_every_residual_unless_told_otherwise():
    """Restricting the set matters when one equation is expected to worsen
    while the one you care about improves."""
    solver = _solver()
    np.testing.assert_array_equal(
        solver.get_active_names(["water", "oil", "well"]), [True]*3)

    solver.linesearch_convergence_names = ["oil"]
    np.testing.assert_array_equal(
        solver.get_active_names(["water", "oil", "well"]),
        [False, True, False])
