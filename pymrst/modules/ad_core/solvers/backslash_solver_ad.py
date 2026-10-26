"""MRST ``ad-core/solvers/BackslashSolverAD.m``: the linear system solved
directly, MATLAB's `A\\b` (here SciPy's sparse LU), its time recorded as
`LinearSolutionTime`. As every solver in this port, `solve_linear_system`
returns the solution and leaves the report to `LinearSolverAD`."""

from __future__ import annotations

import time
from typing import Any, Tuple

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from pymrst.modules.ad_core.solvers.linear_solver_ad import LinearSolverAD


class BackslashSolverAD(LinearSolverAD):
    """MRST's `BackslashSolverAD`."""

    def __init__(self, **options: Any):
        super().__init__()
        for name, value in options.items():
            setattr(self, name, value)

    def solve_linear_system(self, A: Any, b: np.ndarray) -> np.ndarray:
        timer = time.perf_counter()
        result = spla.spsolve(sp.csc_matrix(A), b)
        self.last_iterations = 1
        self.last_solve_report = self.get_solve_report(LinearSolutionTime=time.perf_counter() - timer)
        return result

    def get_description(self) -> Tuple[str, str]:
        return "Matlab \\ operator (mldivide direct solver)", "mldivide" + getattr(self, "id", "")

    solveLinearSystem = solve_linear_system
    getDescription = get_description


__all__ = ["BackslashSolverAD"]


if __name__ == "__main__":
    def test_direct_solve_and_description():
        A = sp.csr_matrix(np.array([[4.0, 1.0], [1.0, 3.0]]))
        solver = BackslashSolverAD()
        assert np.allclose(A @ solver.solve_linear_system(A, np.array([1.0, 2.0])), [1.0, 2.0])
        assert solver.get_description()[1] == "mldivide"
        assert solver.last_solve_report["LinearSolutionTime"] >= 0
        print("ok test_direct_solve_and_description")

    test_direct_solve_and_description()
    print("\nAll tests passed!")
