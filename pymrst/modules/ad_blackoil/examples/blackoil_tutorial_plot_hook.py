"""MRST ``ad-blackoil/examples/blackoilTutorialPlotHook.m``.

In-situ plotting with MRST-AD: SPE1 (`setupSPE1`, `selectModelFromDeck`,
`convertDeckScheduleToMRST`) run by `simulateScheduleAD` with an after-step
function from `getPlotAfterStep` ('plotWell' on, 'plotReservoir' off). After
every report step it refreshes the well-curve viewer and the runtime panel,
whose Abort button stops the run. The linear solver is
`selectLinearSolverAD`'s: SPE1's 900 unknowns get a direct solve.
`blackoil_tutorial_plot_hook` is the tutorial; without `plot` it runs with
no after-step function, which gives the same states.

`build_blackoil_tutorial_plot_hook`/`run_blackoil_tutorial_plot_hook`,
below, are PyMRST's earlier version on the legacy models, kept for their
callers. Its notes follow.

MRST's script, in full::

    [G, rock, fluid, deck, state0] = setupSPE1();
    model    = selectModelFromDeck(G, rock, fluid, deck);
    schedule = convertDeckScheduleToMRST(model, deck);
    fn       = getPlotAfterStep(state0, model, schedule, ...
                                'plotWell', true, 'plotReservoir', false);
    linsolve = selectLinearSolverAD(model);
    [wellSols, states, report] = simulateScheduleAD(state0, model, schedule, ...
        'Verbose', true, 'afterStepFn', fn, 'linearsolver', linsolve);

The previous port ran a *different case* - `simulatorWorkflowExample`'s
two-phase model - through a hand-written loop, and its "hook" was a callback
the loop chose to invoke. Nothing in it went through `getPlotAfterStep`,
`selectModelFromDeck`, `convertDeckScheduleToMRST` or `selectLinearSolverAD`,
which is the entire content of the script: it is a tutorial about *the hook
mechanism*, and the case is SPE1 only because it needs one.

Three things this port keeps that the loop could not:

* **the hook is the simulator's, not the caller's.** `after_step_fn` is passed
  to `simulate_schedule`, which calls it after each accepted report step and
  **stops the run when it returns `ok = False`**. A loop that calls a callback
  itself cannot demonstrate that, because the loop already controls when to
  stop;
* **the model comes from the deck.** `select_model_from_deck_ad` reads SPE1's
  RUNSPEC and returns a three-phase live-oil model; the previous port built a
  two-phase one by hand;
* **the linear solver is chosen by size.** SPE1 is 300 cells and three phases -
  900 unknowns, below `select_linear_solver_ad_ad`'s 10000 threshold - so it picks
  a direct solve. That is the *right* answer for this case, and the point of
  calling the selector rather than hardcoding one is that the same script on a
  field model picks CPR instead.

Plotting is left to the caller: pass `plot_well_fn` to receive MRST's
`wellSols` and cumulative time after every step.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any, Callable, Dict, List, Optional

import numpy as np

from pymrst.modules.ad_core.plotting.get_plot_after_step import (
    get_plot_after_step_ad,
)
from pymrst.modules.ad_core.simulators.nonlinear_solver import NonLinearSolver
from pymrst.modules.ad_core.utils.convert_deck_schedule_to_mrst import (
    convert_deck_schedule_to_mrst_ad,
)
from pymrst.modules.ad_core.utils.schedule import Schedule
from pymrst.modules.ad_core.utils.select_linear_solver_ad import (
    select_linear_solver_ad_ad,
)
from pymrst.modules.ad_core.utils.select_model_from_deck import (
    select_model_from_deck_ad,
)
from pymrst.modules.ad_core.utils.well_sol_from_state import (
    well_sols_from_states,
)
from pymrst.modules.test_suite.dataset_functions.spe1 import setup_spe1_ad



def blackoil_tutorial_plot_hook(plot: bool = True) -> SimpleNamespace:
    """MRST `blackoilTutorialPlotHook`."""
    from pymrst.modules.ad_core.ad_oo.simulate_schedule_ad import simulate_schedule_ad
    from pymrst.modules.ad_core.plotting.get_plot_after_step import get_plot_after_step
    from pymrst.modules.ad_core.utils.convert_deck_schedule_to_mrst import (
        convert_deck_schedule_to_mrst,
    )
    from pymrst.modules.ad_core.utils.select_linear_solver_ad import select_linear_solver_ad
    from pymrst.modules.ad_core.utils.select_model_from_deck import select_model_from_deck
    from pymrst.modules.test_suite.dataset_functions.spe1 import setup_spe1

    G, rock, fluid, deck, state0 = setup_spe1()
    model = select_model_from_deck(G, rock, fluid, deck)
    schedule = convert_deck_schedule_to_mrst(model, deck)
    # Set up plotting function
    fn = None
    if plot:
        import matplotlib.pyplot as plt
        plt.close("all")
        fn = get_plot_after_step(state0, model, schedule, plotWell=True, plotReservoir=False)
        print(fn)
    # Run the simulation with plotting function
    linsolve = select_linear_solver_ad(model)
    wellSols, states, report = simulate_schedule_ad(state0, model, schedule, Verbose=True,
                                                    afterStepFn=fn, LinearSolver=linsolve)
    return SimpleNamespace(G=G, model=model, schedule=schedule, state0=state0, fn=fn,
                           linsolve=linsolve, wellSols=wellSols, states=states,
                           report=report)

@dataclass
class PlotHookResult:
    model: Any
    schedule: Any
    state0: Any
    states: List[Any]
    report: Any
    well_sols: List[List[Dict[str, Any]]]
    linear_solver: Any
    #: One entry per time the hook fired, for a caller that wants to check it.
    hook_calls: List[int] = field(default_factory=list)


def build_blackoil_tutorial_plot_hook():
    """SPE1, its deck-selected model and its deck schedule."""
    G, rock, fluid, deck, state0 = setup_spe1_ad()
    model = select_model_from_deck_ad(G, rock, fluid, deck)
    schedule = convert_deck_schedule_to_mrst_ad(
        model, {"step": deck["SCHEDULE"].step,
                "control": deck["SCHEDULE"].control})
    return model, schedule, state0


def run_blackoil_tutorial_plot_hook(
    nsteps: Optional[int] = None, *,
    plot_well_fn: Optional[Callable] = None,
    stop_when: Optional[Callable[..., bool]] = None,
    verbose: bool = True,
) -> PlotHookResult:
    """Run SPE1 with an after-step hook, as MRST's script does."""
    model, schedule, state0 = build_blackoil_tutorial_plot_hook()
    if nsteps is not None:
        count = min(int(nsteps), int(schedule.n_steps))
        schedule = Schedule(schedule.step_val[:count],
                            schedule.step_control[:count], schedule.controls)
    _seed_well_unknowns(model, state0, schedule.controls[0].W)

    calls: List[int] = []

    def record(well_sols, time):
        calls.append(len(well_sols))
        if plot_well_fn is not None:
            plot_well_fn(well_sols, time)

    # MRST: 'plotWell', true, 'plotReservoir', false.
    fn = get_plot_after_step_ad(state0, model, schedule,
                                plot_well_fn=record, stop_when=stop_when)
    linear_solver = select_linear_solver_ad_ad(model)

    solver = NonLinearSolver(linear_solver=linear_solver)
    solver.verbose = bool(verbose)
    states, report = solver.simulate_schedule(model, state0, schedule,
                                              after_step_fn=fn)
    return PlotHookResult(
        model=model, schedule=schedule, state0=state0, states=states,
        report=report, well_sols=well_sols_from_states(model, states),
        linear_solver=linear_solver, hook_calls=calls,
    )


blackoilTutorialPlotHook = blackoil_tutorial_plot_hook


def _seed_well_unknowns(model: Any, state: Any, wells) -> None:
    """The well primary variables the black-oil models read off the state."""
    wells = list(wells or ())
    phases = [str(name).upper()[0] for name in model.get_phase_names()]
    n = len(wells)
    p_mean = float(np.mean(np.asarray(state["pressure"], dtype=float)))
    rates = {f"q{phase}s": np.zeros(n) for phase in phases}
    bhp = np.zeros(n)
    for index, well in enumerate(wells):
        composition = np.atleast_1d(np.asarray(well.compi, dtype=float))
        if str(well.type).lower() == "bhp":
            bhp[index] = float(well.val)
        else:
            bhp[index] = p_mean + 10e5*int(well.sign or 1)
            for position, phase in enumerate(phases):
                if position < composition.size:
                    rates[f"q{phase}s"][index] = (float(well.val)
                                                  * composition[position])
    for name, values in rates.items():
        state[name] = values
    state["bhp"] = bhp


__all__ = ["PlotHookResult", "blackoilTutorialPlotHook",
           "build_blackoil_tutorial_plot_hook",
           "run_blackoil_tutorial_plot_hook"]


if __name__ == "__main__":
    from pymrst.modules.test_suite.dataset_functions.spe1.setup_spe1 import (
        spe1_data_directory,
    )

    try:
        spe1_data_directory()
    except FileNotFoundError:
        print("skip (SPE1 dataset not installed)")
        raise SystemExit(0)

    MODEL, SCHEDULE, STATE0 = build_blackoil_tutorial_plot_hook()

    def test_the_case_is_spe1_from_its_deck():
        """Not `simulatorWorkflowExample`'s two-phase model, which is what the
        previous port ran."""
        assert MODEL.G.cells.num == 300, MODEL.G.cells.num
        assert MODEL.n_phases == 3, MODEL.n_phases
        assert MODEL.disgas is True, "SPE1 is a live-oil case"
        assert MODEL.vapoil is False, "SPE1's gas is dry"
        print("ok test_the_case_is_spe1_from_its_deck")

    def test_the_schedule_is_the_decks():
        assert SCHEDULE.n_steps == 120, SCHEDULE.n_steps
        assert len(SCHEDULE.controls[0].W) == 2, "one injector, one producer"
        print("ok test_the_schedule_is_the_decks")

    def test_the_linear_solver_is_chosen_by_system_size():
        """900 unknowns is below the 10000 crossover, so a direct solve is
        right here - and the same script on a field model picks CPR."""
        from pymrst.modules.ad_core.solvers.concrete_solvers import (
            BackslashSolverAD,
        )
        solver = select_linear_solver_ad_ad(MODEL)
        assert isinstance(solver, BackslashSolverAD), type(solver)
        print("ok test_the_linear_solver_is_chosen_by_system_size")

    def test_the_hook_fires_once_per_report_step():
        result = run_blackoil_tutorial_plot_hook(3, verbose=False)
        # One seeding call before the run, then one per accepted step.
        assert result.hook_calls == [2, 1, 2, 3], result.hook_calls
        assert len(result.states) == 3
        print("ok test_the_hook_fires_once_per_report_step")

    def test_the_hook_receives_growing_well_solutions():
        seen = []
        run_blackoil_tutorial_plot_hook(
            3, verbose=False,
            plot_well_fn=lambda ws, t: seen.append((len(ws), float(np.max(t)))))
        assert [n for n, _ in seen[1:]] == [1, 2, 3], seen
        assert seen[-1][1] > seen[1][1], "time must advance"
        print("ok test_the_hook_receives_growing_well_solutions")

    def test_the_hook_can_stop_the_run():
        """The veto - the reason this is a hook and not an observer. MRST's own
        `ok` comes from a GUI stop button."""
        result = run_blackoil_tutorial_plot_hook(
            10, verbose=False,
            stop_when=lambda model, states, reports: len(states) >= 2)
        assert len(result.states) == 2, len(result.states)
        assert len(result.report.steps) == 2
        print("ok test_the_hook_can_stop_the_run")

    def test_the_run_produces_well_output():
        result = run_blackoil_tutorial_plot_hook(3, verbose=False)
        assert all(step.converged for step in result.report.steps)
        rows = result.well_sols[-1]
        producer = next(row for row in rows if row["sign"] < 0)
        assert producer["qOs"] < 0.0, producer
        injector = next(row for row in rows if row["sign"] > 0)
        assert injector["qGs"] > 0.0, injector
        print(f"ok test_the_run_produces_well_output "
              f"(oil {producer['qOs']:.3e} m3/s)")

    def test_the_port_runs_spe1_through_the_hook():
        """The hook's well viewer and runtime panel are refreshed every step
        and leave the run as it is without them."""
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from pymrst.params.gravity.gravity import isolated_gravity
        with isolated_gravity():
            hooked = blackoil_tutorial_plot_hook(plot=True)
            plain = blackoil_tutorial_plot_hook(plot=False)
        assert type(hooked.linsolve).__name__ == "BackslashSolverAD"
        assert len(hooked.states) == len(hooked.schedule.step_val) == 120
        assert "mrst-simpanel" in [plt.figure(n).get_label() for n in plt.get_fignums()]
        np.testing.assert_array_equal(hooked.states[-1]["pressure"], plain.states[-1]["pressure"])
        plt.close("all")
        print("ok test_the_port_runs_spe1_through_the_hook")

    test_the_port_runs_spe1_through_the_hook()
    test_the_case_is_spe1_from_its_deck()
    test_the_schedule_is_the_decks()
    test_the_linear_solver_is_chosen_by_system_size()
    test_the_hook_fires_once_per_report_step()
    test_the_hook_receives_growing_well_solutions()
    test_the_hook_can_stop_the_run()
    test_the_run_produces_well_output()
    print("\nAll tests passed!")
