"""Linear-solver diagnostics propagated through Newton reports."""

import numpy as np

from pymrst.modules.ad_core.simulators.nonlinear_solver import NonLinearSolver
from pymrst.modules.ad_core.solvers.concrete_solvers import GMRES_ILUSolverADAD
from pymrst.modules.ad_core.statefunctions.state_function import State
from pymrst.utils.adi import init_variables_adi, value


class _ScalarModel:
    @staticmethod
    def get_equations(state0, state, dt):
        (x,) = init_variables_adi(np.asarray(state["x"], dtype=float))
        return [x - 2.0]

    @staticmethod
    def get_convergence(equations, dt):
        norm = float(np.linalg.norm(value(equations[0])))
        return norm < 1e-12, {"residual": norm}

    @staticmethod
    def update_state(state, increment):
        return State(x=np.asarray(state["x"], dtype=float) + increment)


def test_direct_diagnostic_reaches_step_report_without_changing_solve_contract():
    state = State(x=np.array([0.0]))
    _, report = NonLinearSolver(max_iterations=3).solve_ministep(
        _ScalarModel(), state, state.copy(), 1.0,
    )
    assert report.converged
    assert report.iterations == 1
    assert len(report.linear_solves) == 1
    solve = report.linear_solves[0]
    assert solve.solver == "scipy.sparse.linalg.spsolve"
    assert solve.iterations == 0
    assert solve.converged
    assert solve.relative_residual == 0.0
    assert report.linear_iterations == [0]
    assert report.total_linear_iterations == 0


def test_gmres_iteration_and_true_residual_reach_step_report():
    state = State(x=np.array([0.0]))
    linear = GMRES_ILUSolverADAD(tolerance=1e-12, max_iterations=10)
    _, report = NonLinearSolver(
        max_iterations=3, linear_solver=linear,
    ).solve_ministep(_ScalarModel(), state, state.copy(), 1.0)
    assert report.converged
    assert report.linear_iterations == [1]
    solve = report.linear_solves[0]
    assert solve.solver.startswith("GMRES_ILUSolverADAD")
    assert solve.residual_norm < 1e-12
    assert solve.relative_residual < 1e-12
    assert solve.degrees_of_freedom == 1
