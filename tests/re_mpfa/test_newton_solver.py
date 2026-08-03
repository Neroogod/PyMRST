"""The Richards-equation Newton loop (F0.4 risk list).

Two things here are exactly checkable rather than approximately, and they are
the two that matter.

A **linear** residual must converge in exactly two iterations: Newton lands on
the root with its first update, and the second confirms a zero update. Not
"about two" - exactly two, and to machine precision. A sign error in the
update, a missing negation on the right-hand side, or a solve against the wrong
matrix all break it immediately.

And the **modified-Picard argument contract**: `h_eq(h, h_n0, previous, dt)`
gets the value at *entry to the timestep* as `h_n0` and the *previous iterate*
as `previous`. These coincide on the first iteration and diverge afterwards, so
swapping them produces a solver that still converges - to the solution of a
different time discretisation. Nothing detects that except a test that watches
what the residual is handed.
"""

from __future__ import annotations

import numpy as np
import pytest
import scipy.sparse as sp

from pymrst.modules.re_mpfa.functions.newton_solver import (
    NewtonResult, newton_solver,
)
from pymrst.utils.adi import ADI, init_variables_adi


def _operator(n=5):
    """A tridiagonal operator that is deliberately *not* symmetric.

    The obvious choice - equal off-diagonals - makes the Jacobian its own
    transpose, so solving against `A.T` gives an identical answer and a
    transposed-solve defect passes every test. Confirmed by making that
    mutation against a symmetric operator and watching it survive.
    """
    return sp.diags([2.0, -1.0, -3.0], [0, 1, -1], shape=(n, n)).tocsr()


def _linear_residual(matrix, rhs, log=None):
    """`A h - b`, whose root is known in closed form.

    `matrix @ h` is written as `h.__rmatmul__(matrix)` deliberately: scipy's
    sparse `__matmul__` raises on an unrecognised operand instead of returning
    `NotImplemented`, so Python never gets to offer the reflected operation.
    """
    def residual(h, h_n0, previous, dt):
        if log is not None:
            log.append({"h": h.val.copy(), "h_n0": np.asarray(h_n0).copy(),
                        "previous": np.asarray(previous).copy(), "dt": float(dt)})
        return h.__rmatmul__(matrix) - rhs
    return residual


def test_a_linear_problem_converges_in_exactly_two_iterations():
    matrix, rhs = _operator(), np.arange(1.0, 6.0)
    result = newton_solver(np.zeros(5), _linear_residual(matrix, rhs),
                           0.0, 1.0, 1e-10, 20)
    assert result.iterations == 2
    np.testing.assert_allclose(result.hydraulic_head.val,
                               np.linalg.solve(matrix.toarray(), rhs),
                               rtol=1e-12, atol=1e-12)
    assert result.update_norm < 1e-12


def test_the_residual_receives_the_step_value_and_the_previous_iterate():
    """The modified-Picard contract, watched rather than assumed.

    `h_n0` is the head at entry to the timestep and never changes within it;
    `previous` is the current iterate, and does. They agree on the first
    iteration, which is exactly why swapping them is invisible until the
    second.

    `previous` also equals `h.val` at every call - the two carry the same
    numbers and differ only in that `h` carries derivatives - so an assertion
    tying `previous` to the *preceding* iteration's head would be wrong. It is
    the contrast with the constant `h_n0` that detects a swap.
    """
    matrix, rhs = _operator(), np.arange(1.0, 6.0)
    log = []
    start = np.full(5, 0.25)
    newton_solver(start, _linear_residual(matrix, rhs, log), 0.0, 1.0, 1e-10, 20)

    assert len(log) >= 2, "need a second iteration for the two to differ"
    for call in log:
        np.testing.assert_allclose(call["h_n0"], start, rtol=0, atol=0)
        np.testing.assert_allclose(call["previous"], call["h"], rtol=0, atol=0)

    np.testing.assert_allclose(log[0]["previous"], start, rtol=0, atol=0)
    assert not np.allclose(log[1]["previous"], start), (
        "`previous` still equals the step value on the second iteration, so it "
        "is not tracking the iterate")


def test_the_timestep_is_passed_through_untouched():
    matrix, rhs = _operator(), np.arange(1.0, 6.0)
    log = []
    newton_solver(np.zeros(5), _linear_residual(matrix, rhs, log),
                  10.0, 0.25, 1e-10, 20)
    assert {call["dt"] for call in log} == {0.25}


def test_a_nonlinear_problem_converges_quadratically():
    """Newton's signature. A first-order method reaches the same root and
    would pass any test that only checks the answer."""
    def residual(h, h_n0, previous, dt):
        return h*h - 4.0

    norms = []

    def watched(h, h_n0, previous, dt):
        norms.append(float(np.max(np.abs(h.val - 2.0))))
        return residual(h, h_n0, previous, dt)

    result = newton_solver(np.full(3, 5.0), watched, 0.0, 1.0, 1e-12, 50)
    np.testing.assert_allclose(result.hydraulic_head.val, 2.0, atol=1e-10)
    # Each error is bounded by the square of the one before, up to a constant.
    for before, after in zip(norms, norms[1:]):
        if before > 1e-6:
            assert after <= 2.0*before**2 + 1e-12, f"{before} -> {after}"


def test_the_cumulative_time_advances_by_one_step():
    matrix, rhs = _operator(), np.arange(1.0, 6.0)
    result = newton_solver(np.zeros(5), _linear_residual(matrix, rhs),
                           100.0, 7.5, 1e-10, 20)
    assert result.cumulative_time == pytest.approx(107.5)


def test_failing_to_converge_raises_and_reports_the_update():
    """Returning the unconverged head instead would let a run continue on a
    state that does not satisfy the equation."""
    matrix, rhs = _operator(), np.arange(1.0, 6.0)
    with pytest.raises(RuntimeError, match="did not converge"):
        newton_solver(np.zeros(5), _linear_residual(matrix, rhs),
                      0.0, 1.0, 1e-10, 1)


def test_a_residual_that_is_not_ad_is_refused():
    def plain(h, h_n0, previous, dt):
        return np.asarray(h.val) - 1.0

    with pytest.raises(TypeError, match="ADI"):
        newton_solver(np.zeros(5), plain, 0.0, 1.0, 1e-10, 20)


def test_a_plain_array_is_promoted_and_an_adi_is_not_mutated():
    """The caller's ADI must survive the call, or a retry with a smaller
    timestep silently starts from the failed iterate."""
    matrix, rhs = _operator(), np.arange(1.0, 6.0)

    promoted = newton_solver(np.zeros(5), _linear_residual(matrix, rhs),
                             0.0, 1.0, 1e-10, 20)
    assert isinstance(promoted.hydraulic_head, ADI)

    (given,) = init_variables_adi(np.zeros(5))
    before = given.val.copy()
    newton_solver(given, _linear_residual(matrix, rhs), 0.0, 1.0, 1e-10, 20)
    np.testing.assert_array_equal(given.val, before)


def test_the_previous_iteration_is_the_head_before_the_final_update():
    matrix, rhs = _operator(), np.arange(1.0, 6.0)
    log = []
    result = newton_solver(np.zeros(5), _linear_residual(matrix, rhs, log),
                           0.0, 1.0, 1e-10, 20)
    np.testing.assert_allclose(result.previous_iteration, log[-1]["h"],
                               rtol=0, atol=0)


def test_the_result_still_unpacks_as_mrst_five_outputs():
    """`newtonSolver.m` returns five values and callers unpack them
    positionally; the dataclass keeps that working."""
    matrix, rhs = _operator(), np.arange(1.0, 6.0)
    result = newton_solver(np.zeros(5), _linear_residual(matrix, rhs),
                           0.0, 1.0, 1e-10, 20)
    head, previous, iterations, cumulative, elapsed = result
    assert isinstance(result, NewtonResult)
    assert head is result.hydraulic_head
    np.testing.assert_array_equal(previous, result.previous_iteration)
    assert iterations == result.iterations
    assert cumulative == result.cumulative_time
    assert elapsed == result.elapsed_time
