"""MRST's reduced-system and adjoint solver methods.

Six methods that share one idea: a Newton system is not homogeneous. The cell
equations are the reservoir - big, sparse, elliptic in their pressure part -
and the well and control equations are a handful of rows that couple to a few
cells each. Every method here exploits that split.

* `reduceToSingleVariableType` / `recoverFromSingleVariableType` and
  `solveCellReducedLinearProblem` eliminate the non-cell rows exactly, solve a
  system the size of the grid, then rebuild the eliminated increments;
* `eliminateLocal` exposes the same split as the four blocks `B, C, D, E`;
* `solveAdjointProblem` runs the backward recursion, whose two traps are the
  transpose and the coupling term;
* `applyTwoStagePreconditioner` composes CPR's pressure and ILU stages.

The load-bearing claim throughout is **exactness**: a Schur complement is a
rearrangement, so every reduced solve must reproduce the full solve's answer
for the variables it keeps - not approximate it.
"""

from __future__ import annotations

import numpy as np
import pytest
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from pymrst.modules.ad_core.solvers.concrete_solvers import (
    BackslashSolverAD, GMRES_ILUSolverADAD,
)
# PyMRST's earlier problem class (in place, residual `b`, positions from the
# lookups). MRST's `LinearizedProblem` is linearized_problem.py, checked
# against MRST in examples/validation/validate_ad_core_linearized_problem.py.
from pymrst.modules.ad_core.solvers.linear_solver_ad import LinearizedProblemAD as LinearizedProblem
from pymrst.utils.adi import init_variables_adi

N_CELL = 6
N_WELL = 2


def _problem(seed=0):
    """Two cell equations and one well equation, cells first.

    Cells first is not cosmetic - `recover_from_single_variable_type` requires
    it and says why. The ordering is asserted in its own test.
    """
    rng = np.random.default_rng(seed)

    def block(rows, cols, diag=0.0):
        M = sp.csr_matrix(rng.random((rows, cols)))
        if diag:
            M = M + sp.eye(rows, cols)*diag
        return M

    p, s, q = init_variables_adi(np.full(N_CELL, 1.0e6),
                                 np.full(N_CELL, 0.3),
                                 np.full(N_WELL, 1.0e-3))
    eq_w = (p.__rmatmul__(block(N_CELL, N_CELL, 9.0))
            + s.__rmatmul__(block(N_CELL, N_CELL))
            + q.__rmatmul__(block(N_CELL, N_WELL)) - rng.random(N_CELL))
    eq_o = (p.__rmatmul__(block(N_CELL, N_CELL))
            + s.__rmatmul__(block(N_CELL, N_CELL, 11.0))
            + q.__rmatmul__(block(N_CELL, N_WELL)) - rng.random(N_CELL))
    eq_c = (p.__rmatmul__(block(N_WELL, N_CELL))
            + s.__rmatmul__(block(N_WELL, N_CELL))
            + q.__rmatmul__(block(N_WELL, N_WELL, 7.0)) - rng.random(N_WELL))

    return LinearizedProblem.from_equations(
        [eq_w, eq_o, eq_c], ["cell", "cell", "well"],
        ["water", "oil", "closureWells"], ["pressure", "sW", "qWs"], None, 1.0)


def _full_solution(seed=0):
    A, b = _problem(seed).get_linear_system()
    return np.linalg.solve(A.toarray(), -b)


def _problem_two_wells(seed=3):
    """Two cell equations and *two* well equations.

    A single eliminated equation cannot detect either of the two properties
    recovery depends on: with one block there is no order to reverse and
    nothing to accumulate. Both mutations passed against the one-well problem.
    """
    rng = np.random.default_rng(seed)

    def block(rows, cols, diag=0.0):
        M = sp.csr_matrix(rng.random((rows, cols)))
        return M + sp.eye(rows, cols)*diag if diag else M

    p, s, q1, q2 = init_variables_adi(
        np.full(N_CELL, 1.0e6), np.full(N_CELL, 0.3),
        np.full(N_WELL, 1.0e-3), np.full(N_WELL, 2.0e-3))
    variables = (p, s, q1, q2)
    widths = (N_CELL, N_CELL, N_WELL, N_WELL)

    def equation(rows, diag_at):
        out = -rng.random(rows)
        for j, (v, w) in enumerate(zip(variables, widths)):
            out = out + v.__rmatmul__(
                block(rows, w, 9.0 + j if j == diag_at else 0.0))
        return out

    return LinearizedProblem.from_equations(
        [equation(N_CELL, 0), equation(N_CELL, 1),
         equation(N_WELL, 2), equation(N_WELL, 3)],
        ["cell", "cell", "well", "well"],
        ["water", "oil", "closureA", "closureB"],
        ["pressure", "sW", "qA", "qB"], None, 1.0)


# -- the type mask ----------------------------------------------------------

def test_the_type_mask_is_case_insensitive():
    """MRST compares with `strcmpi` throughout."""
    problem = LinearizedProblem.from_equations(
        ["a", "b", "c"], ["cell", "CELL", "well"], ["w", "o", "g"], ["p"])
    np.testing.assert_array_equal(problem.index_of_type("cell"),
                                  [True, True, False])
    np.testing.assert_array_equal(problem.index_of_type("WELL"),
                                  [False, False, True])


def test_an_unknown_type_selects_nothing():
    assert not _problem().index_of_type("fracture").any()


# -- reduction and recovery -------------------------------------------------

def test_reducing_to_cells_leaves_only_the_cell_equations():
    problem, eliminated = _problem().reduce_to_single_variable_type("cell")
    assert problem.equation_names == ["water", "oil"]
    assert problem.types == ["cell", "cell"]
    assert len(eliminated) == 1


def test_the_reduced_solve_reproduces_the_full_cell_increments():
    """The exactness claim. A Schur complement is a rearrangement, not a
    preconditioner, so this is equality to solver tolerance - not closeness."""
    reference = _full_solution()[:2*N_CELL]

    problem, _ = _problem().reduce_to_single_variable_type("cell")
    A, b = problem.get_linear_system()
    assert A.shape == (2*N_CELL, 2*N_CELL)
    np.testing.assert_allclose(np.linalg.solve(A.toarray(), -b), reference,
                               rtol=1e-8)


def test_recovery_rebuilds_the_eliminated_increment():
    """The well increment must come back, and match the full solve."""
    full = _full_solution()
    original = _problem()
    reduced, eliminated = _problem().reduce_to_single_variable_type("cell")

    A, b = reduced.get_linear_system()
    dx_reduced = np.linalg.solve(A.toarray(), -b)
    blocks = [dx_reduced[:N_CELL], dx_reduced[N_CELL:2*N_CELL]]

    dx = LinearizedProblem.recover_from_single_variable_type(
        reduced, original, blocks, eliminated)

    assert len(dx) == 3
    np.testing.assert_allclose(dx[2], full[2*N_CELL:], rtol=1e-6)


def test_recovery_of_several_blocks_is_exact_and_ordered():
    """Two eliminated equations, which is the smallest case that can detect
    either property recovery relies on.

    Each elimination was formed against the ones already removed, so they must
    be recovered in reverse; and each recovered block joins the set the next
    recovery is computed from. With one block neither shows.
    """
    original = _problem_two_wells()
    A, b = original.get_linear_system()
    full = np.linalg.solve(A.toarray(), -b)

    reduced, eliminated = _problem_two_wells().reduce_to_single_variable_type("cell")
    assert len(eliminated) == 2

    Ar, br = reduced.get_linear_system()
    dx_reduced = np.linalg.solve(Ar.toarray(), -br)
    blocks = [dx_reduced[:N_CELL], dx_reduced[N_CELL:2*N_CELL]]

    dx = LinearizedProblem.recover_from_single_variable_type(
        reduced, original, blocks, eliminated)

    base = 2*N_CELL
    np.testing.assert_allclose(dx[2], full[base:base + N_WELL], rtol=1e-6)
    np.testing.assert_allclose(dx[3], full[base + N_WELL:], rtol=1e-6)


def test_recovery_refuses_a_non_contiguous_keep_set():
    """`recover_vars` addresses the eliminated Jacobian by position, so a kept
    block after an eliminated one is read from the wrong column."""
    problem = LinearizedProblem.from_equations(
        ["a", "b", "c"], ["cell", "well", "cell"], ["w", "x", "o"], ["p"])
    with pytest.raises(ValueError, match="contiguous"):
        LinearizedProblem.recover_from_single_variable_type(
            problem, problem, [np.zeros(1), np.zeros(1)], [])


def test_recovery_refuses_kept_variables_that_do_not_come_first():
    problem = LinearizedProblem.from_equations(
        ["a", "b"], ["well", "cell"], ["x", "w"], ["p"])
    reduced = LinearizedProblem.from_equations(["b"], ["cell"], ["w"], ["p"])
    with pytest.raises(ValueError, match="come first"):
        LinearizedProblem.recover_from_single_variable_type(
            reduced, problem, [np.zeros(1)], [])


def test_the_cell_reduced_solver_matches_a_full_solve():
    """End to end: `solveCellReducedLinearProblem` must give the same
    increments as solving the whole coupled system."""
    full = _full_solution()
    dx, result, report = BackslashSolverAD().solve_cell_reduced_linear_problem(
        _problem())

    assert result.size == 2*N_CELL
    np.testing.assert_allclose(np.concatenate([np.asarray(d).ravel()
                                               for d in dx[:2]]),
                               full[:2*N_CELL], rtol=1e-8)
    assert "SolverTime" in report


# -- the block split --------------------------------------------------------

def test_eliminate_local_eliminates_the_well_equations_of_a_sparse_problem():
    """MRST's `eliminateLocal` on a sparse (ADI) problem: the non-cell
    equations are eliminated (`reduceToVariable`) - the four-block Schur split
    is only for the diagonal backend - and the cell equations remain."""
    problem, eliminated, keep, lsys, _ = \
        BackslashSolverAD().eliminate_local(_problem())

    assert lsys is None and len(eliminated) == 1
    assert list(problem.equation_names) == ["water", "oil"]
    np.testing.assert_array_equal(keep, [True, True, False])


def test_the_reduced_solve_recovers_every_increment():
    """The values, not the shapes: with `reduceToCell` the solve returns
    the full solution, the eliminated well increment included."""
    reference = _full_solution()
    solver = BackslashSolverAD()
    solver.reduceToCell = True
    dx, _, _ = solver.solve_linear_problem(_problem())
    full = np.concatenate([np.ravel(np.asarray(d, dtype=float)) for d in dx])
    np.testing.assert_allclose(full, reference, rtol=1e-8)


def test_nothing_to_eliminate_reports_no_system():
    """`None` for `lsys` keeps "there was no reduction" distinguishable from
    "the reduction produced empty blocks"."""
    problem = LinearizedProblem.from_equations(
        ["a"], ["cell"], ["w"], ["p"])
    _, eliminated, keep, lsys, _ = \
        BackslashSolverAD().eliminate_local(problem)
    assert lsys is None and eliminated == [] and keep.all()


# -- the adjoint ------------------------------------------------------------

def test_the_adjoint_solves_the_transposed_system():
    """The forward step asks how the state responds to a perturbation; the
    adjoint asks which perturbations the objective responds to. Using `A`
    rather than `A^T` answers the wrong question and still returns a vector.
    """
    problem = _problem()
    A, _ = problem.get_linear_system()
    n = A.shape[0]
    rng = np.random.default_rng(5)
    gradient = rng.standard_normal(n)

    _, lam, _ = BackslashSolverAD().solve_adjoint_problem(
        None, problem, None, gradient)

    np.testing.assert_allclose(A.toarray().T @ lam, -gradient, rtol=1e-8)


def test_an_ad_objective_supplies_its_own_gradient():
    problem = _problem()
    n = problem.get_linear_system()[0].shape[0]
    x, = init_variables_adi(np.zeros(n))
    weights = np.arange(1.0, n + 1.0)
    objective = (x*weights).__rmatmul__(sp.csr_matrix(np.ones((1, n))))

    _, lam, _ = BackslashSolverAD().solve_adjoint_problem(
        None, problem, None, objective)

    A = problem.get_linear_system()[0].toarray()
    np.testing.assert_allclose(A.T @ lam, -weights, rtol=1e-8)


def test_the_coupling_term_carries_the_recursion_backwards():
    """`problem_prev` holds dF_{i+1}/dx_i - the only term that carries
    information back in time. Dropping it gives a gradient that is right for
    one step and wrong for a schedule, which reads as a stalling optimiser
    rather than as a bug."""
    curr, prev = _problem(0), _problem(1)
    A = curr.get_linear_system()[0]
    B = prev.get_linear_system()[0]
    n = A.shape[0]
    rng = np.random.default_rng(7)
    gradient = rng.standard_normal(n)
    next_lambda = rng.standard_normal(n)

    _, lam, _ = BackslashSolverAD().solve_adjoint_problem(
        prev, curr, next_lambda, gradient)

    expected = -gradient - B.toarray().T @ next_lambda
    np.testing.assert_allclose(A.toarray().T @ lam, expected, rtol=1e-8)


def test_without_a_next_multiplier_there_is_no_coupling_term():
    """The last step of the schedule: nothing follows it."""
    curr, prev = _problem(0), _problem(1)
    n = curr.get_linear_system()[0].shape[0]
    gradient = np.random.default_rng(9).standard_normal(n)

    _, with_prev, _ = BackslashSolverAD().solve_adjoint_problem(
        prev, curr, None, gradient)
    _, without, _ = BackslashSolverAD().solve_adjoint_problem(
        None, curr, None, gradient)
    np.testing.assert_allclose(with_prev, without, rtol=1e-10)


# -- CPR's two stages -------------------------------------------------------

def _two_stage_setup(n=24, seed=2):
    rng = np.random.default_rng(seed)
    A = sp.random(n, n, density=0.25, random_state=rng, format="csr")
    A = (A + sp.eye(n)*(np.abs(A).sum(axis=1).A1.max() + 1.0)).tocsr()
    p_inx = np.arange(0, n, 2)
    Ap = A.tocsr()[p_inx, :][:, p_inx].tocsr()
    ilu = spla.splu(A.tocsc())
    L = sp.csr_matrix(ilu.L.toarray())
    U = sp.csr_matrix(ilu.U.toarray())
    return A, Ap, L, U, p_inx, rng.standard_normal(n)


def test_the_two_stages_compose_rather_than_overlap():
    """Without the `r - A x` between them, the second stage sees the original
    residual, the corrections overlap, and CPR degrades to roughly ILU alone -
    converging, with no other symptom.

    Checked by construction: applying the second stage to the raw residual
    gives a different vector.
    """
    from pymrst.modules.ad_core.solvers.cpr_solver_ad import CPRSolverADAD

    A, Ap, L, U, p_inx, r = _two_stage_setup()
    solver = CPRSolverADAD()
    elliptic = BackslashSolverAD()

    composed = solver.apply_two_stage_preconditioner(
        r, A, Ap, L, U, p_inx, elliptic_solver=elliptic)

    x = np.zeros_like(r)
    x[p_inx] = elliptic.solve_linear_system(Ap, r[p_inx])
    overlapping = x + spla.spsolve_triangular(
        U, spla.spsolve_triangular(L, r, lower=True), lower=False)

    assert not np.allclose(composed, overlapping)


def test_the_preconditioner_reduces_the_residual():
    """It must be an approximate inverse: applying it to `r` should move
    `A x` towards `r`, or it is not a preconditioner at all."""
    from pymrst.modules.ad_core.solvers.cpr_solver_ad import CPRSolverADAD

    A, Ap, L, U, p_inx, r = _two_stage_setup()
    x = CPRSolverADAD().apply_two_stage_preconditioner(
        r, A, Ap, L, U, p_inx, elliptic_solver=BackslashSolverAD())

    assert np.linalg.norm(A @ x - r) < 0.5*np.linalg.norm(r)


def test_the_scaling_cancels_out():
    """The right-hand-side scaling makes the inner solver's absolute tolerance
    behave relatively; it is divided back out, so an exact inner solve must
    give the same answer whatever the relative tolerance is."""
    from pymrst.modules.ad_core.solvers.cpr_solver_ad import CPRSolverADAD

    A, Ap, L, U, p_inx, r = _two_stage_setup()
    tight = CPRSolverADAD(relative_tolerance=1e-8).apply_two_stage_preconditioner(
        r, A, Ap, L, U, p_inx, elliptic_solver=BackslashSolverAD())
    loose = CPRSolverADAD(relative_tolerance=1e-2).apply_two_stage_preconditioner(
        r, A, Ap, L, U, p_inx, elliptic_solver=BackslashSolverAD())
    np.testing.assert_allclose(tight, loose, rtol=1e-8)


def test_a_missing_elliptic_solver_is_refused():
    from pymrst.modules.ad_core.solvers.cpr_solver_ad import CPRSolverADAD

    A, Ap, L, U, p_inx, r = _two_stage_setup()
    with pytest.raises(ValueError, match="elliptic solver"):
        CPRSolverADAD().apply_two_stage_preconditioner(r, A, Ap, L, U, p_inx)


# -- ILU options ------------------------------------------------------------

def test_the_ilu_options_report_the_settings_in_force():
    solver = GMRES_ILUSolverADAD(ilu_drop_tol=1e-3)
    opts = solver.get_opts_ilu()
    assert opts["droptol"] == 1e-3
    assert set(opts) == {"type", "droptol", "milu", "udiag", "thresh"}


def test_changing_the_drop_tolerance_is_reflected_and_used():
    """The report must track the value the solve actually uses, or it
    describes a configuration that is not running."""
    solver = GMRES_ILUSolverADAD()
    solver.ilu_drop_tol = 5e-2
    assert solver.get_opts_ilu()["droptol"] == 5e-2
    assert solver._scipy_ilu_kwargs()["drop_tol"] == 5e-2


def test_an_inexpressible_option_is_refused_rather_than_ignored():
    """`udiag` has no `spilu` equivalent. Silently dropping it would make a
    solver configured for a singular diagonal behave as though it were not."""
    solver = GMRES_ILUSolverADAD()
    solver.udiagReplacement = True
    with pytest.raises(NotImplementedError, match="udiag"):
        solver._scipy_ilu_kwargs()


def test_the_ilu_solver_still_solves():
    rng = np.random.default_rng(4)
    n = 40
    A = sp.random(n, n, density=0.2, random_state=rng, format="csr")
    A = (A + sp.eye(n)*(np.abs(A).sum(axis=1).A1.max() + 1.0)).tocsr()
    x = rng.standard_normal(n)
    got = GMRES_ILUSolverADAD().solve_linear_system(A, A @ x)
    np.testing.assert_allclose(got, x, rtol=1e-5, atol=1e-8)
