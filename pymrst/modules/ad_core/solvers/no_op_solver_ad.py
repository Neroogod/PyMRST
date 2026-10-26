"""MRST ``ad-core/solvers/NoOpSolverAD.m``: a linear solver that solves
nothing - a zero increment, reported as not converged."""

from __future__ import annotations

from typing import Any

import numpy as np

from pymrst.modules.ad_core.solvers.linear_solver_ad import LinearSolverAD


class NoOpSolverAD(LinearSolverAD):
    """MRST's `NoOpSolverAD`."""

    def solve_linear_system(self, A: Any, b: np.ndarray) -> np.ndarray:
        self.last_iterations = 0
        self.last_solve_report = {"Converged": False}
        return np.zeros(np.shape(b))

    solveLinearSystem = solve_linear_system


__all__ = ["NoOpSolverAD"]


if __name__ == "__main__":
    def test_zero_increment_not_converged():
        solver = NoOpSolverAD()
        assert np.array_equal(solver.solve_linear_system(None, np.ones(3)), np.zeros(3))
        assert solver.last_solve_report == {"Converged": False}
        print("ok test_zero_increment_not_converged")

    test_zero_increment_not_converged()
    print("\nAll tests passed!")
