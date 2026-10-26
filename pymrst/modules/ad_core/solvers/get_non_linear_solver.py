"""MRST ``ad-core/solvers/getNonLinearSolver.m``.

`get_non_linear_solver` is the port: MRST's AD-OO `NonLinearSolver`, its
linear solver from `selectLinearSolverAD`, and a timestep selector for the
chosen strategy. 'iteration' targets 8 Newton iterations per step; 'ds' and
'dsdc' limit the saturation (and component) change. The first step of a
control is ramped up from a tenth of it, or from a day.

`get_non_linear_solver_ad`, below, is PyMRST's earlier factory for the legacy
solver, kept for its callers. Its notes follow.

MRST-compatible ``getNonLinearSolver`` factory (legacy).

This is the public convenience entry point used by MRST examples to assemble a
nonlinear solver, a suitable linear solver, and an optional timestep selector.
"""

from __future__ import annotations

import math
from typing import Any, Iterable, Mapping, Optional

from pymrst.modules.ad_core.simulators.nonlinear_solver import NonLinearSolver
from pymrst.modules.ad_core.solvers.linear_solvers import (
    DirectSolver,
    select_linear_solver,
)
from pymrst.modules.ad_core.timesteps.iteration_count_time_step_selector import (
    IterationCountTimeStepSelector,
)
from pymrst.modules.ad_core.timesteps.state_change_time_step_selector import (
    StateChangeTimeStepSelector,
)
from pymrst.modules.ad_core.timesteps.timestep_selectors import (
    IterationCountTimeStepSelectorAD,
    StateChangeTimeStepSelectorAD,
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


def get_non_linear_solver_ad(
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
        selector = IterationCountTimeStepSelectorAD(target_iteration_count=8)
    elif strategy == "ds":
        selector = StateChangeTimeStepSelectorAD(
            target_props=["s"],
            target_change_abs=[0.2],
            target_iteration_count=10**9,
        )
    elif strategy == "dsdc":
        compositional = (
            model.__class__.__name__ == "ThreePhaseCompositionalModel"
            or bool(getattr(model, "is_compositional", False))
        )
        selector = StateChangeTimeStepSelectorAD(
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
def get_non_linear_solver(model: Any, *, TimestepStrategy: str = "iteration",
                          useCPR: bool = True, LinearSolverArguments: Any = (),
                          **varg: Any):
    """MRST `getNonLinearSolver`."""
    from pymrst.modules.ad_core.ad_oo.nonlinear_solver import (
        NonLinearSolver as AdOoNonLinearSolver,
    )
    from pymrst.modules.ad_core.utils.select_linear_solver_ad import (
        select_linear_solver_ad,
    )
    nonlinear = AdOoNonLinearSolver(**varg)
    nonlinear.LinearSolver = select_linear_solver_ad(model, useCPR=useCPR,
                                                     **_linear_kwargs(LinearSolverArguments))
    strategy = TimestepStrategy.lower()
    if strategy == "none":
        sel = None
    elif strategy == "iteration":
        sel = IterationCountTimeStepSelector(target_iteration_count=8)
    elif strategy == "ds":
        sel = StateChangeTimeStepSelector(target_props=["s"], target_change_abs=[0.2],
                                          target_iteration_count=math.inf)
    elif strategy == "dsdc":
        names, targets = ["s"], [0.2]
        if any(c.__name__ == "ThreePhaseCompositionalModel" for c in type(model).__mro__):
            names, targets = ["s", "components"], [0.2, 0.2]
        sel = StateChangeTimeStepSelector(target_props=names, target_change_abs=targets,
                                          target_iteration_count=math.inf)
    else:
        raise ValueError(f"Unknown timestepping strategy {TimestepStrategy}")
    if sel is not None:
        sel.first_rampup_step_relative = 0.1
        sel.first_rampup_step = 1*DAY
        nonlinear.timeStepSelector = sel
    return nonlinear


getNonLinearSolver = get_non_linear_solver

__all__ = ["get_non_linear_solver", "get_non_linear_solver_ad", "getNonLinearSolver"]


if __name__ == "__main__":
    from types import SimpleNamespace

    def _generic(n_cells):
        model = SimpleNamespace(G=SimpleNamespace(cells=SimpleNamespace(num=n_cells)),
                                isGenericReservoirModel=True, AutoDiffBackend=None,
                                get_component_names=lambda: ["water", "oil", "gas"])
        model.validate_model = lambda: model
        return model

    def test_the_default_targets_eight_iterations_with_a_ramped_first_step():
        nls = get_non_linear_solver(_generic(300), maxIterations=12, useRelaxation=True)
        assert type(nls).__module__.endswith("ad_oo.nonlinear_solver")
        assert nls.maxIterations == 12 and nls.useRelaxation
        sel = nls.timeStepSelector
        assert type(sel).__name__ == "IterationCountTimeStepSelector"
        assert sel.target_iteration_count == 8
        assert sel.first_rampup_step_relative == 0.1 and sel.first_rampup_step == DAY
        assert type(nls.LinearSolver).__name__ == "BackslashSolverAD"
        print("ok test_the_default_targets_eight_iterations_with_a_ramped_first_step")

    def test_ds_limits_the_saturation_change():
        sel = get_non_linear_solver(_generic(300), TimestepStrategy="ds").timeStepSelector
        assert type(sel).__name__ == "StateChangeTimeStepSelector"
        print("ok test_ds_limits_the_saturation_change")

    def test_none_keeps_the_solvers_default_selector():
        nls = get_non_linear_solver(_generic(300), TimestepStrategy="none")
        assert type(nls.timeStepSelector).__name__ != "IterationCountTimeStepSelector"
        print("ok test_none_keeps_the_solvers_default_selector")

    test_the_default_targets_eight_iterations_with_a_ramped_first_step()
    test_ds_limits_the_saturation_change()
    test_none_keeps_the_solvers_default_selector()
    print("\nAll tests passed!")
