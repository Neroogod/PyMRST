"""MRST-compatible ``getNonLinearSolver`` factory.

This is the public convenience entry point used by MRST examples to assemble a
nonlinear solver, a suitable linear solver, and an optional timestep selector.
"""

from __future__ import annotations

from typing import Any, Iterable, Mapping, Optional

from pymrst.modules.ad_core.simulators.nonlinear_solver import NonLinearSolver
from pymrst.modules.ad_core.solvers.linear_solvers import (
    DirectSolver,
    select_linear_solver,
)
from pymrst.modules.ad_core.timesteps.timestep_selectors import (
    IterationCountTimeStepSelector,
    StateChangeTimeStepSelector,
)


DAY = 86_400.0


def _linear_kwargs(arguments: Optional[Any]) -> dict:
    if arguments is None:
        return {}
    if isinstance(arguments, Mapping):
        return dict(arguments)
    if isinstance(arguments, Iterable) and not isinstance(arguments, (str, bytes)):
        values = list(arguments)
        if len(values) % 2:
            raise ValueError("LinearSolverArguments must contain name/value pairs")
        return dict(zip(values[::2], values[1::2]))
    raise TypeError("LinearSolverArguments must be a mapping or name/value sequence")


def get_non_linear_solver(
    model: Any,
    *,
    timestep_strategy: str = "iteration",
    use_cpr: bool = True,
    linear_solver_arguments: Optional[Any] = None,
    **solver_options,
) -> NonLinearSolver:
    """Construct MRST's recommended nonlinear-solver configuration.

    Parameters mirror ``getNonLinearSolver.m``. CamelCase MRST spellings are
    accepted in ``solver_options`` for code translated directly from MATLAB.
    """
    timestep_strategy = solver_options.pop("TimestepStrategy", timestep_strategy)
    use_cpr = solver_options.pop("useCPR", use_cpr)
    linear_solver_arguments = solver_options.pop(
        "LinearSolverArguments", linear_solver_arguments
    )

    aliases = {
        "maxIterations": "max_iterations",
        "maxTimestepCuts": "max_timestep_cuts",
        "useLinesearch": "use_line_search",
        "useRelaxation": "use_relaxation",
        "relaxationMin": "relaxation_min",
        "relaxationIncrement": "relaxation_increment",
        "lineSearchMaxIterations": "line_search_max",
    }
    for old, new in aliases.items():
        if old in solver_options:
            if new in solver_options:
                raise TypeError(f"both {old!r} and {new!r} were supplied")
            solver_options[new] = solver_options.pop(old)

    strategy = str(timestep_strategy).lower()
    if strategy == "none":
        selector = None
    elif strategy == "iteration":
        selector = IterationCountTimeStepSelector(target_iteration_count=8)
    elif strategy == "ds":
        selector = StateChangeTimeStepSelector(
            target_props=["s"],
            target_change_abs=[0.2],
            target_iteration_count=10**9,
        )
    elif strategy == "dsdc":
        compositional = (
            model.__class__.__name__ == "ThreePhaseCompositionalModel"
            or bool(getattr(model, "is_compositional", False))
        )
        selector = StateChangeTimeStepSelector(
            target_props=["s", "components"] if compositional else ["s"],
            target_change_abs=[0.2, 0.2] if compositional else [0.2],
            target_iteration_count=10**9,
        )
    else:
        raise ValueError(f"unknown timestepping strategy {timestep_strategy!r}")

    solver = NonLinearSolver(**solver_options)
    linear_kwargs = _linear_kwargs(linear_solver_arguments)
    solver.linear_solver = (
        select_linear_solver(model, **linear_kwargs)
        if use_cpr
        else DirectSolver()
    )

    if selector is not None:
        selector.first_rampup_step_relative = 0.1
        selector.first_rampup_step = DAY
    solver.timestep_selector = selector
    return solver


# MRST spelling for close translations.
getNonLinearSolver = get_non_linear_solver

__all__ = ["get_non_linear_solver", "getNonLinearSolver"]
