"""Parity tests for MRST ``getNonLinearSolver`` defaults and strategies."""

import pytest

from pymrst.modules.ad_core.solvers import (
    DirectSolver,
    getNonLinearSolver,
    get_non_linear_solver,
)
from pymrst.modules.ad_core.timesteps import (
    IterationCountTimeStepSelector,
    StateChangeTimeStepSelector,
)


class _Cells:
    num = 10


class _Grid:
    cells = _Cells()


class _Model:
    G = _Grid()
    n_phases = 2
    fluid = type("Fluid", (), {"disgas": False, "vapoil": False})()


def test_default_matches_mrst_factory_configuration():
    solver = getNonLinearSolver(_Model())
    assert isinstance(solver.linear_solver, DirectSolver)
    assert isinstance(solver.timestep_selector, IterationCountTimeStepSelector)
    assert solver.timestep_selector.target_iteration_count == 8
    assert solver.timestep_selector.first_rampup_step_relative == 0.1
    assert solver.timestep_selector.first_rampup_step == 86_400.0


@pytest.mark.parametrize(
    ("strategy", "selector"),
    [
        ("none", type(None)),
        ("iteration", IterationCountTimeStepSelector),
        ("ds", StateChangeTimeStepSelector),
        ("dsdc", StateChangeTimeStepSelector),
    ],
)
def test_timestep_strategies(strategy, selector):
    solver = get_non_linear_solver(
        _Model(), timestep_strategy=strategy, use_cpr=False
    )
    assert isinstance(solver.timestep_selector, selector)


def test_mrst_option_spellings_and_solver_options():
    solver = get_non_linear_solver(
        _Model(),
        TimestepStrategy="none",
        useCPR=False,
        maxIterations=7,
        useLinesearch=True,
    )
    assert solver.timestep_selector is None
    assert isinstance(solver.linear_solver, DirectSolver)
    assert solver.max_iterations == 7
    assert solver.use_line_search


def test_unknown_strategy_is_rejected():
    with pytest.raises(ValueError, match="unknown timestepping strategy"):
        get_non_linear_solver(_Model(), timestep_strategy="mystery")
