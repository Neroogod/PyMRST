"""MRST ad-core linear/nonlinear solver layout (`ad-core/solvers/`)."""

from pymrst.modules.ad_core.solvers.linear_solvers import (
    CPR_CROSSOVER_BLOCK_JACOBI,
    CPR_CROSSOVER_ILU,
    HAVE_AMG,
    CPRSolver,
    DirectSolver,
    select_linear_solver,
)
from pymrst.modules.ad_core.solvers.linear_solver_ad import (
    LinearizedProblem,
    LinearSolverAD,
)
from pymrst.modules.ad_core.solvers.concrete_solvers import (
    BackslashSolverAD,
    GMRES_ILUSolverAD,
    NoOpSolverAD,
)
from pymrst.modules.ad_core.solvers.get_non_linear_solver import (
    get_non_linear_solver,
    getNonLinearSolver,
)
from pymrst.modules.ad_core.solvers.handle_linear_solver_ad import (
    HandleLinearSolverAD,
)
from pymrst.modules.ad_core.solvers.cpr_solver_ad import CPRSolverAD

__all__ = [
    "BackslashSolverAD",
    "CPR_CROSSOVER_BLOCK_JACOBI",
    "CPR_CROSSOVER_ILU",
    "CPRSolver",
    "DirectSolver",
    "GMRES_ILUSolverAD",
    "HAVE_AMG",
    "LinearSolverAD",
    "LinearizedProblem",
    "NoOpSolverAD",
    "get_non_linear_solver",
    "getNonLinearSolver",
    "HandleLinearSolverAD",
    "CPRSolverAD",
    "select_linear_solver",
]
