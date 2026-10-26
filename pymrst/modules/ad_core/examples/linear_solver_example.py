"""Runnable native-solver comparison corresponding to MRST ``linearSolverExample``."""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter

import numpy as np
import scipy.sparse as sp

from pymrst.modules.ad_core.solvers import (
    BackslashSolverAD,
    CPRSolverADAD,
    GMRES_ILUSolverADAD,
)


@dataclass(frozen=True)
class LinearSolverExampleResult:
    solver: str
    solution: np.ndarray
    relative_residual: float
    relative_error: float
    iterations: int
    elapsed: float


def linear_solver_example(
    *,
    size: int = 120,
    seed: int = 0,
    tolerance: float = 1e-10,
) -> list[LinearSolverExampleResult]:
    """Solve one deterministic sparse system with PyMRST's AD solvers.

    Optional AGMG/AMGCL wrappers from the MATLAB example require external
    binaries and are intentionally absent; native direct, GMRES-ILU, and CPR
    compatibility paths are compared against the same known solution.
    """

    if size < 2:
        raise ValueError("size must be at least two")
    rng = np.random.default_rng(seed)
    matrix = sp.random(
        size, size, density=min(12.0 / size, 0.5),
        random_state=rng, format="csr",
    )
    matrix = matrix + sp.diags(
        np.asarray(np.abs(matrix).sum(axis=1)).ravel() + 2.0
    )
    expected = rng.standard_normal(size)
    rhs = np.asarray(matrix @ expected).ravel()
    solvers = (
        BackslashSolverAD(),
        GMRES_ILUSolverADAD(
            tolerance=tolerance, max_iterations=300, ilu_drop_tol=1e-6
        ),
        CPRSolverADAD(tolerance=tolerance, max_iterations=300),
    )
    results = []
    rhs_norm = max(float(np.linalg.norm(rhs)), np.finfo(float).eps)
    expected_norm = max(
        float(np.linalg.norm(expected)), np.finfo(float).eps
    )
    for solver in solvers:
        start = perf_counter()
        solution = solver.solve(matrix, rhs)
        elapsed = perf_counter() - start
        results.append(
            LinearSolverExampleResult(
                solver=type(solver).__name__,
                solution=solution,
                relative_residual=float(
                    np.linalg.norm(matrix @ solution - rhs) / rhs_norm
                ),
                relative_error=float(
                    np.linalg.norm(solution - expected) / expected_norm
                ),
                iterations=int(solver.last_iterations),
                elapsed=float(elapsed),
            )
        )
    return results


linearSolverExample = linear_solver_example

__all__ = [
    "LinearSolverExampleResult",
    "linear_solver_example",
    "linearSolverExample",
]
