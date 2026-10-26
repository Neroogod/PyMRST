"""MRST ``ad-blackoil/examples/blackoilTimeIntegrationExample.m``: one
black-oil case (SPE1, or SPE9) simulated with three time discretizations -
fully implicit, explicit with a CFL-limited time step, and adaptive implicit,
which treats each cell explicitly or implicitly by its local CFL number.

MRST's script: `initEclipseProblemAD` with `'TimestepStrategy', 'none'`,
`setTimeDiscretization` for the two variants, three packed problems sharing
the one nonlinear solver, simulated in turn; then the well curves, AIM's
fraction of implicit cells and the ministeps each took. `plot=False` leaves
out the figures; `nsteps` (a PyMRST addition) runs the first report steps
only, under its own output name so it never mixes with a full run's.

`run_blackoil_time_integration_example_ad`, below, is PyMRST's earlier
version on the legacy models, kept for its callers.
"""

from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any, List, Optional, Tuple

import numpy as np

NAMES = ("Fully-Implicit", "Explicit", "Adaptive-Implicit (AIM)")


def pack_time_integration_problems(name: str = "spe1",
                                   nsteps: Optional[int] = None) -> Tuple[List[Any], Any]:
    """The script up to `problems = {implicit, explicit, aim}`."""
    from pymrst.modules.ad_core.initialization.init_eclipse_problem_ad import (
        init_eclipse_problem_ad,
    )
    from pymrst.modules.ad_core.simulators.sim_runner.pack_simulation_problem import (
        pack_simulation_problem,
    )
    from pymrst.modules.ad_core.utils.set_time_discretization import set_time_discretization

    if name == "spe1":
        from pymrst.modules.test_suite.dataset_functions.spe1.setup_spe1 import setup_spe1
        G, rock, fluid, deck, state = setup_spe1()
    elif name == "spe9":
        from pymrst.modules.test_suite.dataset_functions.spe9.setup_spe9 import setup_spe9
        G, rock, fluid, deck, state = setup_spe9()
    else:
        raise ValueError(f"Case {name} not supported")
    state0, model, schedule, nls = init_eclipse_problem_ad(deck, TimestepStrategy="none")
    base = name
    if nsteps is not None:
        schedule.step_val = np.asarray(schedule.step_val)[:nsteps]
        schedule.step_control = np.asarray(schedule.step_control)[:nsteps]
        base = f"{name}_first{nsteps}"
    # Fully-implicit: the model as it is
    implicit = pack_simulation_problem(state0, model, schedule, base,
                                       NonLinearSolver=nls, Name=NAMES[0])
    # Explicit: a time-step restriction from the CFL condition in each cell
    model_explicit = set_time_discretization(model, "explicit")
    explicit = pack_simulation_problem(state0, model_explicit, schedule, base,
                                       NonLinearSolver=nls, Name=NAMES[1])
    # Adaptive implicit: explicit where the local CFL allows it
    model_aim = set_time_discretization(model, "adaptive-implicit")
    aim = pack_simulation_problem(state0, model_aim, schedule, base,
                                  NonLinearSolver=nls, Name=NAMES[2])
    return [implicit, explicit, aim], schedule


def blackoil_time_integration_example(name: str = "spe1", plot: bool = True,
                                      nsteps: Optional[int] = None) -> SimpleNamespace:
    """MRST `blackoilTimeIntegrationExample` (`name` is the script's variable)."""
    from pymrst.modules.ad_core.simulators.sim_runner.get_multiple_packed_simulator_outputs import (
        get_multiple_packed_simulator_outputs,
    )
    from pymrst.modules.ad_core.simulators.sim_runner.simulate_packed_problem import (
        simulate_packed_problem,
    )
    from pymrst.modules.ad_core.utils.get_report_ministeps import get_report_ministeps

    problems, schedule = pack_time_integration_problems(name, nsteps)
    simulate_packed_problem(problems)
    ws, states, reports, names, T = get_multiple_packed_simulator_outputs(problems)[:5]
    imp_frac = np.array([np.sum(x["implicit"])/np.size(x["implicit"]) for x in states[2]])
    ministeps = [get_report_ministeps(r) for r in reports]
    if plot:
        _plot(ws, T, names, imp_frac, ministeps)
    return SimpleNamespace(problems=problems, schedule=schedule, ws=ws, states=states,
                           reports=reports, names=names, T=T, imp_frac=imp_frac,
                           ministeps=ministeps)


def _plot(ws: Any, T: Any, names: Any, imp_frac: np.ndarray, ministeps: List[np.ndarray]) -> None:
    import matplotlib.pyplot as plt

    from pymrst.modules.ad_core.plotting.plot_well_sols import plot_well_sols
    from pymrst.utils.units import DAY
    plot_well_sols(ws, T, datasetnames=names)
    plt.figure()
    plt.plot(100*imp_frac)
    plt.title("Number of implicit cells in AIM")
    plt.ylabel("Percentage of implicit cells")
    plt.xlabel("Step index")
    plt.figure()
    for dt, marker in zip(ministeps, ("o", ".", "x")):
        plt.plot(np.arange(1, dt.size + 1)/dt.size, dt/DAY, marker=marker)
    plt.xlabel("Timestep length [day]")
    plt.legend(names)


blackoilTimeIntegrationExample = blackoil_time_integration_example


# ---------------------------------------------------------------------------
# PyMRST's earlier version.


from pymrst.modules.ad_blackoil.examples.spe1.blackoil_tutorial_spe1 import (
    SPE1TutorialResult,
    _collect_history,
    _well_indices,
    setup_spe1_tutorial,
)
from pymrst.modules.ad_core import NonLinearSolver, SimulationReport
from pymrst.modules.ad_core.utils import set_time_discretization_ad


@dataclass
class TimeIntegrationDiagnostics:
    report_steps: np.ndarray
    ministeps: np.ndarray
    iterations: np.ndarray
    cuts: np.ndarray
    implicit_fraction: np.ndarray


@dataclass
class TimeIntegrationVariant:
    result: SPE1TutorialResult
    diagnostics: TimeIntegrationDiagnostics


def _representative_wells(model) -> tuple:
    """One producer and one injector to report rates for.

    `_well_indices` in the SPE1 tutorial requires *exactly* one of each, which
    is right for SPE1 and wrong for anything else - SPE9 has 26 wells and the
    check refused the whole comparison. The diagnostics only need a
    representative pair to plot, so the first of each is taken.

    Which pair is arbitrary and the comparison does not depend on it: the
    three temporal discretizations are run on the same model and the same
    wells, so any consistent choice compares like with like. Reporting a
    *sum* over wells would not - a producer and an injector cancel.
    """
    import numpy as _np
    signs = _np.array([w.sign for w in model.wells], dtype=int)
    producers = _np.flatnonzero(signs < 0)
    injectors = _np.flatnonzero(signs > 0)
    if not producers.size or not injectors.size:
        raise ValueError(
            "the time-integration comparison needs at least one producer and "
            "one injector to report rates for")
    return int(producers[0]), int(injectors[0])


def _setup_case(case: str):
    """`(deck, model, state0)` for the named benchmark.

    SPE9 used to be refused here, on the stated grounds that its deck, VFP
    tables and schedule importer were pending. They are not - `setup_spe9`
    builds all three and `simulate_spe9_ad` runs on them - so the refusal had
    outlived its reason, the same way the compositional deck branch had.

    The two cases differ only in how the model is built: SPE1's tutorial
    returns `(deck, model, state0)` directly, SPE9's setup additionally
    returns a schedule this comparison does not use, because it steps the
    model itself to keep the three temporal discretizations on identical
    timesteps.
    """
    if case == "spe9":
        from pymrst.modules.ad_blackoil.examples.spe9.simulate_spe9 import (
            setup_spe9_ad,
        )
        deck, model, _schedule, state0 = setup_spe9_ad()
        return deck, model, state0
    return setup_spe1_tutorial()


def _run_spe1_variant(mode: str, nsteps: int | None,
                      case: str = "spe1") -> TimeIntegrationVariant:
    deck, model, state0 = _setup_case(case)
    set_time_discretization_ad(model, mode)
    timesteps = np.asarray(deck["tsteps"], dtype=float)
    if nsteps is not None:
        timesteps = timesteps[:int(nsteps)]

    solver = NonLinearSolver(max_iterations=30, use_line_search=True)
    report = SimulationReport()
    states = []
    elapsed = []
    state = state0.copy()
    total = 0.0
    actual_steps = []
    implicit_fraction = []
    for step_no, dt in enumerate(timesteps, start=1):
        state, step = solver.solve_timestep(model, state, float(dt))
        report.steps.append(step)
        if not step.converged:
            raise RuntimeError(
                f"{mode} {case.upper()} report step {step_no} failed")
        states.append(state.copy())
        total += float(dt)
        elapsed.append(total)
        actual_steps.extend(step.ministeps or [float(dt)])
        implicit_fraction.append(
            float(np.mean(state["implicit"])) if "implicit" in state else
            (1.0 if mode == "fully-implicit" else 0.0)
        )

    producer, injector = _representative_wells(model)
    result = SPE1TutorialResult(
        deck=deck, model=model, initial_state=state0, states=states,
        report=report, history=_collect_history(states),
        time=np.asarray(elapsed), producer_index=producer,
        injector_index=injector,
    )
    diagnostics = TimeIntegrationDiagnostics(
        report_steps=timesteps,
        ministeps=np.asarray(actual_steps),
        iterations=np.asarray([s.iterations for s in report.steps], dtype=int),
        cuts=np.asarray([s.cuts for s in report.steps], dtype=int),
        implicit_fraction=np.asarray(implicit_fraction),
    )
    return TimeIntegrationVariant(result, diagnostics)


def run_blackoil_time_integration_example_ad(
    name: str = "spe1", nsteps: int | None = 10,
) -> dict[str, TimeIntegrationVariant]:
    """Run all three temporal discretizations on the selected benchmark."""
    case = name.lower()
    if case not in ("spe1", "spe9"):
        raise ValueError(f"unknown black-oil time-integration case: {name!r}")
    return {
        "Fully-Implicit": _run_spe1_variant("fully-implicit", nsteps, case),
        "Explicit": _run_spe1_variant("explicit", nsteps, case),
        "Adaptive-Implicit (AIM)": _run_spe1_variant("adaptive-implicit",
                                                     nsteps, case),
    }


__all__ = [
    "NAMES", "TimeIntegrationDiagnostics", "TimeIntegrationVariant",
    "blackoilTimeIntegrationExample", "blackoil_time_integration_example",
    "pack_time_integration_problems", "run_blackoil_time_integration_example_ad",
]


if __name__ == "__main__":
    def test_the_three_discretizations_on_spe1():
        out = blackoil_time_integration_example("spe1", plot=False, nsteps=3)
        assert list(out.names) == list(NAMES) and all(len(s) == 3 for s in out.states)
        assert np.all(out.imp_frac <= 1) and out.ministeps[1].size >= 3
        print("ok test_the_three_discretizations_on_spe1")

    test_the_three_discretizations_on_spe1()
    print("\nAll tests passed!")
