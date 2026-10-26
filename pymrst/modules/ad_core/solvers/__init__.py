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
    LinearizedProblemAD,
    LinearSolverAD,
)
from pymrst.modules.ad_core.solvers.concrete_solvers import (
    BackslashSolverAD,
    GMRES_ILUSolverADAD,
    NoOpSolverAD,
)
from pymrst.modules.ad_core.solvers.gmres_ilu_solver_ad import GMRES_ILUSolverAD
from pymrst.modules.ad_core.solvers.get_non_linear_solver import (
    get_non_linear_solver,
    get_non_linear_solver_ad,
    getNonLinearSolver,
)
from pymrst.modules.ad_core.solvers.handle_linear_solver_ad import (
    HandleLinearSolverAD,
)
from pymrst.modules.ad_core.solvers.cpr_solver_ad import CPRSolverADAD

__all__ = [
    "BackslashSolverAD",
    "CPR_CROSSOVER_BLOCK_JACOBI",
    "CPR_CROSSOVER_ILU",
    "CPRSolver",
    "DirectSolver",
    "GMRES_ILUSolverAD",
    "GMRES_ILUSolverADAD",
    "HAVE_AMG",
    "LinearSolverAD",
    "LinearizedProblem",
    "LinearizedProblemAD",
    "NoOpSolverAD",
    "get_non_linear_solver",
    "get_non_linear_solver_ad",
    "getNonLinearSolver",
    "HandleLinearSolverAD",
    "CPRSolverADAD",
    "select_linear_solver",
]
