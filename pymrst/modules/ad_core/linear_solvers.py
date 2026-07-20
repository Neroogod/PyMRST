"""Backward-compatible import path for the AD-core linear solvers.

The implementation lives in `ad_core.solvers.linear_solvers`, mirroring
MRST's `ad-core/solvers/` (`CPRSolverAD.m`, `LinearSolverAD.m`).
"""

from pymrst.modules.ad_core.solvers.linear_solvers import (
    CPR_CROSSOVER_BLOCK_JACOBI,
    CPR_CROSSOVER_ILU,
    HAVE_AMG,
    CPRSolver,
    DirectSolver,
    select_linear_solver,
)

__all__ = [
    "CPR_CROSSOVER_BLOCK_JACOBI",
    "CPR_CROSSOVER_ILU",
    "CPRSolver",
    "DirectSolver",
    "HAVE_AMG",
    "select_linear_solver",
]
