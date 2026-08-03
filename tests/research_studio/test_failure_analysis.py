"""Synthetic and real-solver failure classifier tests."""

import numpy as np

from pymrst.modules.ad_core.simulators.nonlinear_solver import NonLinearSolver
from pymrst.modules.ad_core.statefunctions.state_function import State
from pymrst.research_studio import FailureKind, analyze_residual_history
from pymrst.utils.adi import ad_exp, init_variables_adi, value


def test_stall_floor_and_worst_owner_are_identified():
    history = [
        {"water": 1.0, "well": 0.1},
        {"water": 0.995, "well": 0.08},
        {"water": 0.992, "well": 0.07},
        {"water": 0.991, "well": 0.06},
    ]
    result = analyze_residual_history(history)
    assert FailureKind.STALL in result.classifications
    assert FailureKind.RESIDUAL_FLOOR in result.classifications
    assert result.worst_residual == "water" and result.worst_iteration == 0
    assert result.median_convergence_ratio > 0.99


def test_divergence_oscillation_nonfinite_and_linear_failure_are_distinct():
    divergent = analyze_residual_history([
        {"mass": 1.0}, {"mass": 20.0}, {"mass": 5.0},
        {"mass": 30.0}, {"mass": 4.0}, {"mass": 40.0},
    ])
    assert FailureKind.DIVERGENCE in divergent.classifications
    assert FailureKind.OSCILLATION in divergent.classifications
    failed = analyze_residual_history([
        {"mass": 1.0}, {"mass": {"nonfinite": "nan"},
                        "linear_failure": {"nonfinite": "inf"}},
    ])
    assert FailureKind.NONFINITE in failed.classifications
    assert FailureKind.LINEAR_FAILURE in failed.classifications


def test_control_chattering_is_detected_from_reversals():
    result = analyze_residual_history(
        [{"control": 1.0}]*5,
        control_history=["rate", "bhp", "rate", "bhp", "rate"],
    )
    assert FailureKind.CONTROL_CHATTERING in result.classifications


def test_real_nonconvergent_newton_history_is_classifiable():
    class NoRoot:
        @staticmethod
        def get_equations(state0, state, dt):
            (x,) = init_variables_adi(state["x"])
            return [ad_exp(x)+1.0]

        @staticmethod
        def get_convergence(equations, dt):
            residual = float(np.max(np.abs(value(equations[0]))))
            return residual < 1e-12, {"residual": residual}

        @staticmethod
        def update_state(state, increment):
            updated = state.copy()
            updated["x"] = np.clip(state["x"]+increment, -20.0, 20.0)
            return updated

    _, report = NonLinearSolver(max_iterations=5).solve_ministep(
        NoRoot(), State(x=np.array([1.0])), State(x=np.array([1.0])), 1.0,
    )
    result = analyze_residual_history(report.residuals, converged=report.converged)
    assert not report.converged
    assert result.classifications
    assert result.worst_residual == "residual"
