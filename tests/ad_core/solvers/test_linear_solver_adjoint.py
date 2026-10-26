"""The adjoint half of the linear-solver preparation stack (ad_core port).

The adjoint solves `A' lam = b`, so every step of the forward stack has a
mirrored partner: rows and columns swap roles, and with them **left scaling
swaps with right**, and the **variable ordering swaps with the equation
ordering**. Recovery changes too - `s = E^-T (h - C^T x)` against the forward
`s = E^-1 (h - D x)`.

Getting a mirror wrong produces a multiplier vector of the right shape and the
wrong values, and therefore an adjoint gradient that is quietly incorrect. That
is the failure this subsystem exists to prevent, and it is why the tests here
compare against a *directly transposed solve* rather than against each other.

`test_the_forward_and_adjoint_variants_are_not_interchangeable` is the one that
gives the rest their meaning: if the two were the same, the mirroring would be
decoration.
"""

from __future__ import annotations

import numpy as np
import pytest
import scipy.sparse as sp

from pymrst.modules.ad_core.solvers.linear_solver_ad import LinearSolverAD

N = 10


def _system(seed=0):
    """Asymmetric on purpose - a symmetric matrix is its own transpose, and
    every test below would pass without the adjoint code being right."""
    rng = np.random.default_rng(seed)
    A = sp.csr_matrix(rng.random((N, N)) + np.eye(N)*6)
    b = rng.random(N)
    return A, b, np.linalg.solve(A.toarray().T, b)


def _solver(**kwargs):
    solver = LinearSolverAD()
    for key, value in kwargs.items():
        setattr(solver, key, value)
    return solver


def test_the_test_matrix_is_asymmetric():
    """Guards every comparison below from being vacuous."""
    A, _, _ = _system()
    assert not np.allclose(A.toarray(), A.toarray().T)


# -- reduction --------------------------------------------------------------

def test_adjoint_reduction_and_recovery_reproduce_the_transposed_solve():
    A, b, direct = _system()
    solver = _solver(keep_number=4)

    reduced_A, reduced_b, system, _ = solver.reduce_linear_system_adjoint(A, b)
    lam = np.linalg.solve(reduced_A.toarray().T, reduced_b)
    np.testing.assert_allclose(
        solver.recover_linear_system_adjoint(lam, system), direct, rtol=1e-9)


def test_adjoint_recovery_uses_the_transposed_blocks():
    """`E^-T (h - C^T x)`. Using `D` instead of `C^T`, or dropping the
    transpose on `E`, gives a vector that solves nothing - and the shape is
    identical either way, so only the values can tell."""
    A, b, direct = _system()
    solver = _solver(keep_number=4)
    reduced_A, reduced_b, system, _ = solver.reduce_linear_system_adjoint(A, b)
    lam = np.linalg.solve(reduced_A.toarray().T, reduced_b)

    recovered = solver.recover_linear_system_adjoint(lam, system)
    assert recovered.shape == direct.shape
    np.testing.assert_allclose(recovered[4:], direct[4:], rtol=1e-9)


# -- scaling ----------------------------------------------------------------

def test_the_adjoint_undoes_left_scaling_where_the_forward_undoes_right():
    """The mirror. In the transposed system the left factor acts on the
    unknown, so it is the one that must be undone."""
    A, b, direct = _system()
    solver = _solver(apply_left_diagonal_scaling=True)

    scaled_A, scaled_b, scaling, _ = solver.apply_scaling_adjoint(A, b)
    lam = np.linalg.solve(scaled_A.toarray().T, scaled_b)
    assert not np.allclose(lam, direct), "left scaling must change the unknown"
    np.testing.assert_allclose(solver.undo_scaling_adjoint(lam, scaling),
                               direct, rtol=1e-9)


def test_adjoint_right_scaling_leaves_the_multipliers_alone():
    A, b, direct = _system()
    solver = _solver(apply_right_diagonal_scaling=True)
    scaled_A, scaled_b, scaling, _ = solver.apply_scaling_adjoint(A, b)
    lam = np.linalg.solve(scaled_A.toarray().T, scaled_b)
    np.testing.assert_allclose(solver.undo_scaling_adjoint(lam, scaling),
                               direct, rtol=1e-9)


def test_asking_for_both_adjoint_scalings_is_refused():
    A, b, _ = _system()
    solver = _solver(apply_left_diagonal_scaling=True,
                     apply_right_diagonal_scaling=True)
    with pytest.raises(ValueError, match="both left and right"):
        solver.apply_scaling_adjoint(A, b)


# -- ordering ---------------------------------------------------------------

def test_the_adjoint_deorders_by_equation_where_the_forward_uses_variables():
    """In the transposed system the unknown is indexed by equations."""
    A, b, direct = _system()
    rng = np.random.default_rng(3)
    permutation = rng.permutation(N)
    solver = _solver(variable_ordering=permutation,
                     equation_ordering=permutation)

    ordered_A, ordered_b, order, _ = solver.reorder_linear_system(A, b)
    lam = np.linalg.solve(ordered_A.toarray().T, ordered_b)
    np.testing.assert_allclose(
        solver.deorder_linear_system_adjoint(lam, order), direct, rtol=1e-9)


def test_the_adjoint_reads_the_equation_ordering_not_the_variable_one():
    """The two orderings must *differ*, or nothing distinguishes them.

    An earlier version of this test used the same permutation for both, so
    swapping which one the adjoint reads changed nothing and the mutation
    survived. Here they are deliberately different, and the expected answer is
    computed from the equation ordering alone.
    """
    variables = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 0])
    equations = np.arange(N)[::-1]
    assert not np.array_equal(variables, equations), "they must differ"

    solver = _solver(variable_ordering=variables, equation_ordering=equations)
    x = np.arange(float(N))
    order = {"variable_ordering": variables, "equation_ordering": equations}

    expected = np.empty_like(x)
    expected[equations] = x[:equations.size]
    np.testing.assert_array_equal(
        solver.deorder_linear_system_adjoint(x, order), expected)

    # and it must NOT match what the forward variant produces
    assert not np.array_equal(solver.deorder_linear_system(x, order),
                              solver.deorder_linear_system_adjoint(x, order))


def test_the_adjoint_deorder_does_nothing_without_an_equation_ordering():
    solver = _solver(variable_ordering=np.arange(N)[::-1])
    x = np.arange(float(N))
    np.testing.assert_array_equal(
        solver.deorder_linear_system_adjoint(x, {"equation_ordering": None}), x)


# -- the mirror is real -----------------------------------------------------

def test_the_forward_and_adjoint_variants_are_not_interchangeable():
    """What gives every test above its meaning.

    If the forward and adjoint paths produced the same answer, the mirroring
    would be decoration and none of these tests would be checking anything.
    Each pair is asserted to *disagree* on the same input.
    """
    A, b, _ = _system()

    left = _solver(apply_left_diagonal_scaling=True)
    forward_A, forward_b, _, _ = left.apply_diagonal_scaling(A, b)
    adjoint_A, adjoint_b, _, _ = left.apply_scaling_adjoint(A, b)
    assert not np.allclose(forward_b, adjoint_b), (
        "forward left scaling scales b; the adjoint does not")

    ordering = {"variable_ordering": np.arange(N)[::-1],
                "equation_ordering": None}
    solver = _solver(variable_ordering=np.arange(N)[::-1])
    x = np.arange(float(N))
    assert not np.array_equal(solver.deorder_linear_system(x, ordering),
                              solver.deorder_linear_system_adjoint(x, ordering))


# -- eliminating whole equations -------------------------------------------

def test_equations_can_be_eliminated_down_to_the_ones_kept():
    from pymrst.modules.ad_core.solvers.linear_solver_ad import LinearizedProblem
    from pymrst.utils.adi import init_variables_adi

    a11 = sp.csr_matrix(np.eye(3)*4.0)
    a12 = sp.csr_matrix(np.eye(3)*1.0)
    p, s = init_variables_adi(np.ones(3), np.ones(3))
    eq1 = p.__rmatmul__(a11) + s.__rmatmul__(a12) - np.ones(3)
    eq2 = p.__rmatmul__(a12) + s.__rmatmul__(a11) - np.ones(3)

    problem = LinearizedProblem.from_equations(
        [eq1, eq2], ["cell", "cell"], ["water", "oil"], ["pressure", "sW"])
    solver = LinearSolverAD()
    problem, eliminated = solver.reduce_to_variable(problem, [True, False])

    assert problem.equation_names == ["water"]
    assert len(eliminated) == 1, "the eliminated equation must be kept"


def _two_wells(seed=3):
    """Two cell equations and two well equations, cells first."""
    import scipy.sparse as sp
    from pymrst.modules.ad_core.solvers.linear_solver_ad import LinearizedProblem
    from pymrst.utils.adi import init_variables_adi
    rng = np.random.default_rng(seed)
    nc, nw = 6, 2
    variables = init_variables_adi(np.full(nc, 1e6), np.full(nc, 0.3),
                                   np.full(nw, 1e-3), np.full(nw, 2e-3))
    widths = (nc, nc, nw, nw)

    def equation(rows, diag_at):
        out = -rng.random(rows)
        for j, (v, w) in enumerate(zip(variables, widths)):
            M = sp.csr_matrix(rng.random((rows, w)))
            if j == diag_at:
                M = M + sp.eye(rows, w)*(9.0 + j)
            out = out + v.__rmatmul__(M)
        return out

    return LinearizedProblem.from_equations(
        [equation(nc, 0), equation(nc, 1), equation(nw, 2), equation(nw, 3)],
        ["cell", "cell", "well", "well"], ["water", "oil", "wA", "wB"],
        ["pressure", "sW", "qA", "qB"], None, 1.0)


def test_recovering_puts_each_increment_back_in_its_own_slot():
    """MRST's `recoverResult`: every eliminated increment rebuilt from its
    equation (`recoverVars`), the last eliminated first, each in its own slot -
    a misplaced slot would give one well's increment to the other."""
    from pymrst.modules.ad_core.solvers.concrete_solvers import BackslashSolverAD
    A, b = _two_wells().get_linear_system()
    reference = np.linalg.solve(A.toarray(), b)      # MRST's b is minus the residual
    solver = BackslashSolverAD()
    solver.reduceToCell = True
    dx, _, _ = solver.solve_linear_problem(_two_wells())
    offsets = np.cumsum([0, 6, 6, 2, 2])
    for k, d in enumerate(dx):
        np.testing.assert_allclose(np.ravel(np.asarray(d, dtype=float)),
                                   reference[offsets[k]:offsets[k + 1]], rtol=1e-8)


