"""MRST's `LinearizedProblem` equation layer (Phase 2).

MRST's `LinearizedProblem` holds the *equations* - AD residuals, each carrying
one Jacobian block per primary variable - and assembles `(A, b)` on demand.
PyMRST's took `(A, b)` already assembled. Both views are useful and both now
exist, but they are different enough that overloading one constructor on
argument types is a trap: `PhysicalModel.setup_linearized_problem` fell into
it, passing MRST's `(eqs, types, names, primaryVars, state, dt)` into the
matrix constructor, so the equations landed in `A` and the type strings in `b`.

It raised on the first call - and nothing called it, so it shipped in Phase 1
and was found only when the ledger drove a test through this path. That is what
`test_the_model_builds_a_problem_from_equations` is for.

The load-bearing numerical claim is `eliminate_variable`: a Schur complement is
an exact rearrangement, not an approximation, so the reduced system must give
the *same* answer for the variables it keeps.
"""

from __future__ import annotations

import copy

import numpy as np
import pytest
import scipy.sparse as sp

# PyMRST's earlier problem class (in place, residual `b`, positions from the
# lookups). MRST's `LinearizedProblem` is linearized_problem.py, checked
# against MRST in examples/validation/validate_ad_core_linearized_problem.py.
from pymrst.modules.ad_core.solvers.linear_solver_ad import LinearizedProblemAD as LinearizedProblem
from pymrst.utils.adi import init_variables_adi

N = 6


def _blocks(seed=0):
    """A two-variable system that is diagonally dominant, so it is solvable."""
    rng = np.random.default_rng(seed)
    return (sp.csr_matrix(rng.random((N, N)) + np.eye(N)*5),
            sp.csr_matrix(rng.random((N, N))),
            sp.csr_matrix(rng.random((N, N))),
            sp.csr_matrix(rng.random((N, N)) + np.eye(N)*7),
            rng.random(N), rng.random(N))


def _problem(seed=0):
    """Two coupled equations in two primary variables.

    `__rmatmul__` is called explicitly: scipy's sparse `@` raises on an
    unrecognised operand rather than returning `NotImplemented`, so Python
    never offers the reflected operation.
    """
    a11, a12, a21, a22, r1, r2 = _blocks(seed)
    p, s = init_variables_adi(np.full(N, 1.0e6), np.full(N, 0.3))
    eq1 = p.__rmatmul__(a11) + s.__rmatmul__(a12) - r1
    eq2 = p.__rmatmul__(a21) + s.__rmatmul__(a22) - r2
    return LinearizedProblem.from_equations(
        [eq1, eq2], ["cell", "cell"], ["water", "oil"], ["pressure", "sW"],
        None, 1.0)


# -- the defect this file exists to have caught ----------------------------

def test_the_model_builds_a_problem_from_equations():
    """`setup_linearized_problem` shipped broken in Phase 1 because nothing
    exercised it. This is the test that would have."""
    from pymrst.modules.ad_core.models.physical_model import PhysicalModel

    problem = PhysicalModel().setup_linearized_problem(
        ["eq1", "eq2"], ["cell", "well"], ["water", "oil"],
        ["pressure", "bhp"], None, 1.0)

    assert problem.equations == ["eq1", "eq2"]
    assert problem.types == ["cell", "well"]
    assert problem.equation_names == ["water", "oil"]
    assert problem.primary_variables == ["pressure", "bhp"]
    assert problem.num_eq() == 2
    assert problem.dt == 1.0


# -- assembly ---------------------------------------------------------------

def test_the_system_assembles_block_row_by_equation_block_column_by_variable():
    problem = _problem()
    A, b = problem.get_linear_system()
    assert A.shape == (2*N, 2*N)
    assert b.size == 2*N

    a11, a12, _, _, r1, _ = _blocks()
    np.testing.assert_allclose(A[:N, :N].toarray(), a11.toarray())
    np.testing.assert_allclose(A[:N, N:].toarray(), a12.toarray())


def test_the_assembled_system_solves_the_equations_it_came_from():
    problem = _problem()
    A, b = problem.get_linear_system()
    dx = np.linalg.solve(A.toarray(), -b)
    np.testing.assert_allclose(A @ dx, -b, rtol=1e-10)


def test_changing_the_equations_drops_a_stale_matrix():
    """An assembled `A` that outlives its equations is the worst state to be
    in: it still solves, and it solves the previous problem."""
    problem = _problem()
    problem.get_linear_system()
    assert problem.A is not None

    problem.append_equations("eq3", "well", "gas")
    assert problem.A is None, "appending must invalidate the assembled system"


# -- the exact-rearrangement claim -----------------------------------------

def test_eliminating_a_variable_preserves_the_solution_exactly():
    """A Schur complement is a rearrangement, not a preconditioner.

    The reduced system must give the *same* pressure increment as the full
    one - not a nearby one - so the comparison is relative and tight.
    """
    full = _problem()
    A, b = full.get_linear_system()
    reference = np.linalg.solve(A.toarray(), -b)[:N]

    reduced = _problem()
    reduced.eliminate_variable("oil")
    A2, b2 = reduced.get_linear_system()
    assert A2.shape == (N, N)

    solution = np.linalg.solve(A2.toarray(), -b2)
    np.testing.assert_allclose(solution, reference, rtol=1e-8)


def test_elimination_removes_the_equation_and_its_variable():
    problem = _problem()
    eliminated = problem.eliminate_variable("oil")

    assert problem.equation_names == ["water"]
    assert problem.primary_variables == ["pressure"]
    assert problem.types == ["cell"]
    assert eliminated is not None
    assert problem.num_eq() == 1


def test_eliminating_by_index_matches_eliminating_by_name():
    by_name = _problem()
    by_name.eliminate_variable("oil")
    by_index = _problem()
    by_index.eliminate_variable(1)

    np.testing.assert_allclose(by_name.get_linear_system()[0].toarray(),
                               by_index.get_linear_system()[0].toarray())


def test_eliminating_an_unknown_equation_is_refused():
    with pytest.raises(KeyError, match="temperature"):
        _problem().eliminate_variable("temperature")


# -- bookkeeping ------------------------------------------------------------

def test_a_mismatched_equation_set_is_refused_at_construction():
    """A count mismatch becomes a silent misalignment later - equation three
    reported under equation four's name."""
    with pytest.raises(ValueError, match="inconsistent counts"):
        LinearizedProblem.from_equations(["a", "b"], ["cell"], ["x", "y"], ["p"])


def test_a_single_equation_is_accepted_without_wrapping():
    problem = LinearizedProblem.from_equations("eq", "cell", "water", "pressure")
    assert problem.equations == ["eq"]
    assert problem.primary_variables == ["pressure"]


def test_equations_can_be_added_at_either_end():
    problem = LinearizedProblem.from_equations(["b"], ["cell"], ["oil"], ["p"])
    problem.append_equations("c", "well", "gas")
    problem.prepend_equations("a", "cell", "water")
    assert problem.equation_names == ["water", "oil", "gas"]
    assert problem.types == ["cell", "cell", "well"]


def test_equations_of_a_type_are_counted_case_insensitively():
    problem = LinearizedProblem.from_equations(
        ["a", "b", "c"], ["cell", "CELL", "well"], ["w", "o", "g"], ["p"])
    assert problem.count_of_type("cell") == 2
    assert problem.count_of_type("well") == 1
    assert problem.count_of_type("nothing") == 0


def test_lookups_return_none_rather_than_a_wrong_index():
    problem = _problem()
    assert problem.index_of_equation_name("oil") == 1
    assert problem.index_of_equation_name("gas") is None
    assert problem.index_of_primary_variable("sW") == 1
    assert problem.index_of_primary_variable("rs") is None


def test_reordering_moves_names_and_types_with_the_equations():
    """Reordering the equations alone would leave every name attached to the
    wrong residual - and the system would still assemble and solve."""
    problem = LinearizedProblem.from_equations(
        ["a", "b", "c"], ["cell", "well", "cell"], ["w", "o", "g"], ["p"])
    problem.reorder_equations([2, 0, 1])
    assert problem.equations == ["c", "a", "b"]
    assert problem.equation_names == ["g", "w", "o"]
    assert problem.types == ["cell", "cell", "well"]


def test_a_reorder_of_the_wrong_length_is_refused():
    problem = LinearizedProblem.from_equations(["a", "b"], ["cell"]*2,
                                                ["w", "o"], ["p"])
    with pytest.raises(ValueError, match="2 equations"):
        problem.reorder_equations([0])


def test_popping_an_increment_returns_the_block_and_the_rest():
    problem = _problem()
    dx = [np.full(N, 1.0), np.full(N, 2.0)]

    block, rest = problem.pop_increment(dx, "sW")
    np.testing.assert_allclose(block, 2.0)
    assert len(rest) == 1

    absent, unchanged = problem.pop_increment(dx, "rs")
    assert absent is None and len(unchanged) == 2

    block, rest, reduced = problem.pop_increment(dx, "sW", nargout=3)
    assert reduced.primary_variables == ["pressure"] and reduced is not problem
    assert problem.primary_variables == ["pressure", "sW"]


def test_each_equation_reports_its_row_count():
    problem = _problem()
    np.testing.assert_array_equal(problem.get_equation_var_num(), [N, N])
