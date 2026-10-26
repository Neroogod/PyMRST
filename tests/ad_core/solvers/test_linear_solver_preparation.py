"""MRST `LinearSolverAD` preparation and recovery (Phase 2).

A production linear solve is rarely `A \\ b`. It is: reduce the system to the
block you can solve well, scale it so the solver sees sensible magnitudes,
reorder it to suit the preconditioner, solve, then undo all three in reverse.

Each step has an *exact* inverse, and that is what makes this testable without
a tolerance argument: reduce-then-recover, scale-then-unscale and
reorder-then-deorder must all return the original solution to machine
precision. Getting one of those inverses subtly wrong gives an answer that is
close enough to look plausible and wrong enough to matter, which is the failure
mode these tests exist to remove.

Everything here is inert at its defaults - no reduction, no scaling, no
reordering - so a solver that ignores it behaves exactly as before.
"""

from __future__ import annotations

import numpy as np
import pytest
import scipy.sparse as sp

from pymrst.modules.ad_core.solvers.linear_solver_ad import LinearSolverAD

N = 10


def _system(seed=0):
    """Diagonally dominant, so it is solvable and well conditioned."""
    rng = np.random.default_rng(seed)
    A = sp.csr_matrix(rng.random((N, N)) + np.eye(N)*6)
    b = rng.random(N)
    return A, b, np.linalg.solve(A.toarray(), b)


def _solver(**kwargs):
    solver = LinearSolverAD()
    for key, item in kwargs.items():
        setattr(solver, key, item)
    return solver


# -- the defaults change nothing -------------------------------------------

def test_every_preparation_step_is_inert_by_default():
    A, b, _ = _system()
    solver = _solver()

    A1, b1, system, _ = solver.reduce_linear_system(A, b)
    assert A1 is A and system == {}

    A2, b2, scaling, _ = solver.apply_diagonal_scaling(A, b)
    assert A2 is A and scaling == {}

    A3, b3, order, _ = solver.reorder_linear_system(A, b)
    assert order["variable_ordering"] is None


# -- reduction --------------------------------------------------------------

def test_reduction_keeps_the_solution_of_the_block_it_keeps():
    """A Schur complement is exact. The reduced system must give the *same*
    first `keep_number` entries, not nearby ones."""
    A, b, full = _system()
    solver = _solver(keep_number=4)

    reduced_A, reduced_b, _, _ = solver.reduce_linear_system(A, b)
    assert reduced_A.shape == (4, 4)
    np.testing.assert_allclose(
        np.linalg.solve(reduced_A.toarray(), reduced_b), full[:4], rtol=1e-10)


def test_recovery_rebuilds_the_eliminated_block():
    A, b, full = _system()
    solver = _solver(keep_number=4)

    reduced_A, reduced_b, system, _ = solver.reduce_linear_system(A, b)
    x = np.linalg.solve(reduced_A.toarray(), reduced_b)
    np.testing.assert_allclose(solver.recover_linear_system(x, system), full,
                               rtol=1e-10)


def test_a_keep_number_covering_everything_reduces_nothing():
    A, b, _ = _system()
    reduced_A, _, system, _ = _solver(keep_number=N).reduce_linear_system(A, b)
    assert reduced_A is A and system == {}


def test_an_initial_guess_is_truncated_with_the_system():
    A, b, _ = _system()
    _, _, _, x0 = _solver(keep_number=4).reduce_linear_system(
        A, b, x0=np.arange(float(N)))
    assert x0.size == 4


# -- scaling ----------------------------------------------------------------

def test_left_scaling_leaves_the_solution_alone():
    """Left scaling multiplies both sides, so the unknown is unchanged -
    which is why undoing it would be wrong."""
    A, b, full = _system()
    solver = _solver(apply_left_diagonal_scaling=True)

    scaled_A, scaled_b, scaling, _ = solver.apply_diagonal_scaling(A, b)
    x = np.linalg.solve(scaled_A.toarray(), scaled_b)
    np.testing.assert_allclose(x, full, rtol=1e-10)
    np.testing.assert_allclose(solver.undo_scaling(x, scaling), full, rtol=1e-10)


def test_right_scaling_changes_the_unknown_and_must_be_undone():
    """Right scaling solves for `M^-1 x`, so the raw answer is *not* the
    solution until it is mapped back."""
    A, b, full = _system()
    solver = _solver(apply_right_diagonal_scaling=True)

    scaled_A, scaled_b, scaling, _ = solver.apply_diagonal_scaling(A, b)
    y = np.linalg.solve(scaled_A.toarray(), scaled_b)
    assert not np.allclose(y, full), "right scaling must change the unknown"
    np.testing.assert_allclose(solver.undo_scaling(y, scaling), full, rtol=1e-10)


def test_asking_for_both_scalings_is_refused():
    A, b, _ = _system()
    solver = _solver(apply_left_diagonal_scaling=True,
                     apply_right_diagonal_scaling=True)
    with pytest.raises(ValueError, match="both left and right"):
        solver.apply_diagonal_scaling(A, b)


def test_a_zero_on_the_diagonal_is_left_unscaled_rather_than_infinite():
    """One bad row would otherwise turn the whole system into NaN."""
    A = sp.csr_matrix(np.diag([2.0, 0.0, 4.0]))
    M = LinearSolverAD().get_diagonal_inverse(A)
    np.testing.assert_allclose(M.diagonal(), [0.5, 1.0, 0.25])


def test_the_diagonal_inverse_uses_magnitudes():
    """A negative pivot must not flip the sign of its row."""
    A = sp.csr_matrix(np.diag([-4.0, 2.0]))
    np.testing.assert_allclose(
        LinearSolverAD().get_diagonal_inverse(A).diagonal(), [0.25, 0.5])


def test_a_non_square_matrix_is_refused():
    with pytest.raises(ValueError, match="square"):
        LinearSolverAD().get_diagonal_inverse(sp.csr_matrix(np.ones((2, 3))))


# -- reordering -------------------------------------------------------------

def test_reordering_and_deordering_return_the_original_solution():
    A, b, full = _system()
    rng = np.random.default_rng(3)
    permutation = rng.permutation(N)
    solver = _solver(variable_ordering=permutation,
                     equation_ordering=permutation)

    ordered_A, ordered_b, order, _ = solver.reorder_linear_system(A, b)
    x = np.linalg.solve(ordered_A.toarray(), ordered_b)
    np.testing.assert_allclose(solver.deorder_linear_system(x, order), full,
                               rtol=1e-10)


def test_a_short_ordering_leaves_the_tail_in_place():
    """Truncating instead would silently drop equations."""
    A, b, _ = _system()
    solver = _solver(variable_ordering=np.array([2, 0, 1]))
    _, _, order, _ = solver.reorder_linear_system(A, b)
    np.testing.assert_array_equal(order["variable_ordering"][:3], [2, 0, 1])
    np.testing.assert_array_equal(order["variable_ordering"][3:],
                                  np.arange(3, N))


def test_an_ordering_may_be_computed_from_the_system():
    A, b, _ = _system()
    solver = _solver(variable_ordering=lambda A, b: np.argsort(b))
    _, _, order, _ = solver.reorder_linear_system(A, b)
    np.testing.assert_array_equal(order["variable_ordering"], np.argsort(b))


# -- odds and ends ----------------------------------------------------------

def test_a_preconditioner_may_be_a_matrix_or_a_callable():
    solver = LinearSolverAD()
    x = np.array([2.0, 4.0])
    np.testing.assert_allclose(
        solver.preconditioner_inverse(sp.csr_matrix(np.diag([2.0, 4.0])), x),
        [1.0, 1.0])
    np.testing.assert_allclose(
        solver.preconditioner_inverse(lambda v: v*10, x), [20.0, 40.0])
    np.testing.assert_allclose(solver.preconditioner_inverse(None, x), x)


def test_the_solve_report_separates_preparation_from_the_solve():
    """Reduction and scaling can cost more than the solve they were meant to
    speed up, and a single total hides that."""
    report = LinearSolverAD().get_solve_report(Iterations=7)
    assert report["Iterations"] == 7
    for key in ("PreparationTime", "LinearSolutionTime", "PostProcessTime",
                "SolverTime", "Residual", "Converged"):
        assert key in report


def test_an_initial_guess_function_is_used_when_supplied():
    solver = LinearSolverAD()
    assert solver.get_initial_guess(object()) is None
    solver.initial_guess_fun = lambda problem: np.zeros(3)
    np.testing.assert_array_equal(solver.get_initial_guess(object()),
                                  np.zeros(3))
