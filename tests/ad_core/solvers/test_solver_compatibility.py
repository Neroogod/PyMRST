import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from pymrst.modules.ad_core.solvers import (
    CPRSolverAD,
    HandleLinearSolverAD,
    LinearizedProblem,
)


def _system():
    matrix = sp.csr_matrix([[4.0, -1.0], [-1.0, 3.0]])
    exact = np.array([2.0, -1.0])
    return matrix, matrix @ exact, exact


def test_handle_linear_solver_wraps_callable():
    matrix, rhs, exact = _system()
    solver = HandleLinearSolverAD(lambda A, b: spla.spsolve(A.tocsc(), b))
    assert np.allclose(solver.solve(matrix, rhs), exact)
    assert "function handle" in solver.get_description()


def test_cpr_solver_ad_direct_metadata_free_path_and_problem_sign():
    matrix, rhs, exact = _system()
    solver = CPRSolverAD(tolerance=1e-12)
    assert np.allclose(solver.solve(matrix, rhs), exact)
    problem = LinearizedProblem(matrix, -rhs)
    assert np.allclose(solver.solve_linear_problem(problem), exact)
