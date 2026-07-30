"""MRST ``CPRSolverAD`` compatibility surface for PyMRST's CPR solver."""

from __future__ import annotations

from pymrst.modules.ad_core.solvers.linear_solver_ad import (
    LinearizedProblem,
    LinearSolverAD,
)
from pymrst.modules.ad_core.solvers.linear_solvers import CPRSolver


class CPRSolverAD(LinearSolverAD):
    """Two-stage CPR/GMRES solver using PyMRST's validated implementation."""

    def __init__(self, *, relative_tolerance=1e-4, tolerance=1e-8,
                 max_iterations=200, **options):
        super().__init__(
            tolerance=tolerance, max_iterations=max_iterations,
            verbose=bool(options.pop("verbose", False)),
        )
        self.relativeTolerance = float(relative_tolerance)
        self.backend = CPRSolver(
            rtol=tolerance, maxiter=max_iterations,
            **options,
        )

    def solve_linear_system(self, A, b):
        # Without model/state metadata CPR cannot form its pressure weighting;
        # the backend deliberately performs an exact sparse direct solve.
        result = self.backend.solve(A, b)
        self.last_iterations = self.backend.last_iterations
        return result

    def solve(self, jac, rhs, model=None, state=None, dt=None, **kwargs):
        result = self.backend.solve(
            jac, rhs, model=model, state=state, dt=dt
        )
        self.last_iterations = self.backend.last_iterations
        return result

    def solve_linear_problem(self, problem: LinearizedProblem,
                             model=None, state=None):
        return self.solve(
            problem.A, -problem.b, model=model,
            state=problem.state if state is None else state,
            dt=problem.dt,
        )

    def get_description(self):
        return "Constrained pressure residual (CPR) with GMRES."


__all__ = ["CPRSolverAD"]
