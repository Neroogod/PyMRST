"""MRST ``ad-core/plotting/getPlotAfterStep.m``.

`get_plot_after_step` is the port. It opens a well-curve viewer
(`plotWellSols`, seeded with two copies of the initial well solution and
its reservoir-rate statistics) and a reservoir viewer (`plotToolbar` over
`{state0; state0}`), either of which `plotWell`/`plotReservoir` can turn
off, and returns the after-step function `simulateScheduleAD` calls. After
each control step that function feeds the computed states to the reservoir
viewer and their well solutions (against the converged ministep times, or
the step indices) to the well viewer, then shows the
`simulationRuntimePanel`, whose Abort button makes it return `ok = False`.

`get_plot_after_step_ad`, below, is PyMRST's earlier hook, kept for its
callers. Its notes follow.

MRST's version opens two figures - a well-curve plot and a reservoir toolbar -
and returns a closure the simulator calls after every report step, which
refreshes them and returns `ok`. The figures are MATLAB; the **hook** is not,
and the hook is what the example is about:

    fn = getPlotAfterStep(state0, model, schedule, 'plotWell', true, ...);
    simulateScheduleAD(state0, model, schedule, 'afterStepFn', fn);

Two properties carry over exactly and are what this port preserves:

* **the signature and the veto.** The simulator calls
  `(model, states, reports, solver, schedule, simtime)` and reads back
  `(model, states, reports, solver, ok)`. Returning `ok = False` *stops the
  run*. MRST's own `ok` comes from a GUI stop button; a hook that could only
  watch would be an observer, and `NonLinearSolver` already has one of those;
* **what it is handed.** Every accepted state and report so far, plus the
  cumulative simulated time - not just the latest step. That is what lets a
  monitor plot a curve rather than a point, and what lets a stopping rule look
  at a trend.

Rendering is left to the caller: `plot_well_fn` and `plot_reservoir_fn` take a
callable each, and passing neither gives a hook that only enforces the stopping
rule - a useful thing on its own.

They are spelled with the `_fn` suffix rather than as MRST's `plotWell` and
`plotReservoir` because `plot_well` is itself a real drawing routine in
`pymrst.plotting`. A parameter of that name shadows it, and
`test_no_selftest_opens_a_blocking_plot_window` - which decides what can open a
window by tracing which functions *reach* a drawing primitive - then reads this
module as one that draws.
"""

from __future__ import annotations

from typing import Any, Callable, List, Optional, Sequence, Tuple

import numpy as np



def get_plot_after_step(state0: Any, model: Any, schedule: Any, *, plotWell: bool = True,
                        plotReservoir: bool = True, view: Any = None, wells: Any = None,
                        useTimesteps: bool = False, plotWellSolArgs: Sequence[Any] = (),
                        **extra: Any) -> Callable:
    """MRST `fn = getPlotAfterStep(state0, model, schedule, ...)`."""
    from copy import copy

    G = model.G
    W = list(schedule.controls[0].W or [])
    hwell = inject_well = None
    if W and plotWell:
        from pymrst.modules.ad_core.ad_oo.facility_model import FacilityModel
        from pymrst.modules.ad_core.plotting.plot_well_sols import plot_well_sols
        from pymrst.modules.ad_core.utils.init_well_sol_ad import init_well_sol_ad
        ws = init_well_sol_ad(W, model, state0)
        nc = sum(np.asarray(w.cells).size for w in W)
        d = np.ones(nc)
        sources = {"phaseVolume": [d for on in model.get_active_phases() if on]}
        model = copy(model)                # MATLAB changes its own copy
        model.FacilityModel = FacilityModel(model).setup_wells(W)
        ws = model.FacilityModel.set_well_sol_statistics(ws, sources)
        ws0 = [ws, ws]
        if useTimesteps:
            step_val = np.asarray(schedule.step_val, dtype=float)
            hwell, inject_well = plot_well_sols(ws0, step_val[[0, 0]], *plotWellSolArgs)
        else:
            hwell, inject_well = plot_well_sols(ws0, *plotWellSolArgs)
    inject_data = hdata = None
    if plotReservoir:
        hdata, inject_data = _reservoir_viewer(G, [state0, state0], view, wells, extra)

    def after_step(model, states, reports, solver, schedule, simtime):
        import matplotlib.pyplot as plt
        computed = [k for k, x in enumerate(states) if x is not None]
        ctrl_computed = [k for k, x in enumerate(reports) if x is not None]
        current = computed[-1] if computed else None
        st = [states[k] for k in computed]
        rep = [reports[k] for k in ctrl_computed]
        simtime = np.asarray(simtime, dtype=float)[ctrl_computed]
        if inject_data is not None and hdata.is_open():
            inject_data(model.G, st, current)
        if inject_well is not None and plt.fignum_exists(hwell.number):
            plt.figure(hwell.number)
            ws = [x["wellSol"] for x in st]
            inject_well([ws], _get_timesteps(rep))
        from pymrst.modules.ad_core.plotting.simulation_runtime_panel import (
            simulation_runtime_panel,
        )
        ok = simulation_runtime_panel(model, states, reports, solver, schedule, simtime)
        return model, states, reports, solver, ok

    return after_step


getPlotAfterStep = get_plot_after_step


def _get_timesteps(reports: Sequence[Any]) -> np.ndarray:
    """MRST `getTimesteps`: the cumulative times of the converged ministeps."""
    T = [r["Timestep"] for report in reports for r in report["StepReports"]
         if r["Converged"]]
    return np.cumsum(T)


class _ReservoirViewer:
    """`plotToolbar(G, states)` whose states are replaced as the run goes,
    MRST's `injData(G, states, current)`."""

    def __init__(self, G: Any, states: Sequence[Any], view: Any, wells: Any, extra: dict):
        from pymrst.plotting.extensions.plot_toolbar import PlotToolbar
        self.toolbar = PlotToolbar(G, list(states), **extra)
        pl = self.toolbar.pl
        if view is not None:
            pl.view_vector(_view_vector(view))
        if wells:
            from pymrst.plotting.plot_well import plot_well
            plot_well(G, wells, plotter=pl)
        if self.toolbar.qt or self.toolbar.off_screen:
            self.toolbar.show(block=False)
        else:
            pl.show(interactive_update=True, auto_close=False)

    def is_open(self) -> bool:
        return not getattr(self.toolbar.pl, "_closed", False)

    def inject(self, G: Any, states: Sequence[Any], current: int) -> None:
        from pymrst.plotting.extensions.plot_toolbar import normalize_data
        tb = self.toolbar
        tb.fields = normalize_data(list(states), G.cells.num)
        tb.n_steps = max(v.shape[0] for v in tb.fields.values())
        tb.set_step(min(int(current), tb.n_steps - 1))
        if not (tb.qt or tb.off_screen):
            tb.pl.update()


def _reservoir_viewer(G: Any, states: Sequence[Any], view: Any, wells: Any, extra: dict):
    viewer = _ReservoirViewer(G, states, view, wells, extra)
    return viewer, viewer.inject


def _view_vector(view: Any) -> Tuple[float, float, float]:
    """MATLAB `view([az, el])` as a camera direction."""
    az, el = np.deg2rad(np.asarray(view, dtype=float).ravel()[:2])
    return (np.sin(az)*np.cos(el), -np.cos(az)*np.cos(el), np.sin(el))


def get_plot_after_step_ad(state0: Any, model: Any, schedule: Any, *,
                        plot_well_fn: Optional[Callable] = None,
                        plot_reservoir_fn: Optional[Callable] = None,
                        stop_when: Optional[Callable[..., bool]] = None,
                        ) -> Callable:
    """Build an after-step function for `simulate_schedule`.

    ``plot_well_fn(well_sols, time)`` and ``plot_reservoir_fn(G, states, current)``
    are called with what MRST hands its own two renderers. ``stop_when(model,
    states, reports)`` ends the run when it returns ``True``.
    """
    G = model.G
    wells = _wells(schedule, model)
    well_sols0 = None
    if wells and plot_well_fn is not None:
        from pymrst.modules.ad_core.utils.init_well_sol_ad import (
            init_well_sol_ad,
        )
        # MRST seeds the well plot with two copies of the initial well
        # solution, so the axes exist before the first step is taken.
        well_sols0 = init_well_sol_ad(wells, model, state0)
        plot_well_fn([well_sols0, well_sols0], np.zeros(2))
    if plot_reservoir_fn is not None:
        plot_reservoir_fn(G, [state0, state0], 0)

    def after_step(model, states, reports, solver, schedule, simtime):
        computed = [state for state in states if state is not None]
        finished = [report for report in reports if report is not None]
        current = len(computed) - 1
        times = np.asarray(simtime, dtype=float)[:len(finished)]

        if plot_reservoir_fn is not None and computed:
            plot_reservoir_fn(model.G, computed, current)
        if plot_well_fn is not None and computed:
            # From the state, not from `state["wellSol"]`: those rows are
            # `init_well_sol_ad`'s seeds and the solve does not update them.
            from pymrst.modules.ad_core.utils.well_sol_from_state import (
                well_sols_from_states,
            )
            well_sols = well_sols_from_states(model, computed)
            plot_well_fn(well_sols, times[:len(well_sols)])

        ok = True
        if stop_when is not None:
            ok = not bool(stop_when(model, computed, finished))
        return model, states, reports, solver, ok

    return after_step




def _wells(schedule: Any, model: Any) -> List[Any]:
    controls = getattr(schedule, "controls", None) or getattr(
        schedule, "control", None)
    if controls:
        wells = getattr(controls[0], "W", None)
        if wells:
            return list(wells)
    return list(getattr(getattr(model, "facility", model), "wells", ()) or ())


__all__ = ["getPlotAfterStep", "get_plot_after_step", "get_plot_after_step_ad"]


if __name__ == "__main__":
    from types import SimpleNamespace

    from pymrst.gridprocessing import cart_grid, compute_geometry
    from pymrst.modules.ad_blackoil.models import GenericBlackOilModelAD
    from pymrst.modules.ad_core import State
    from pymrst.modules.ad_core.simulators.nonlinear_solver import (
        NonLinearSolver,
    )
    from pymrst.modules.ad_core.utils.schedule import simple_schedule
    from pymrst.modules.ad_props import init_simple_adi_fluid
    from pymrst.params import add_well, make_rock
    from pymrst.utils.units import BARSA, CENTI, DARCY, DAY, MILLI, POISE

    def _case(n=6):
        g = compute_geometry(cart_grid([n, n, 1], [100.0, 100.0, 10.0]))
        rock = make_rock(g, 100*MILLI*DARCY, 0.3)
        fluid = init_simple_adi_fluid(
            mu=np.array([1.0, 5.0])*CENTI*POISE, rho=[1000.0, 800.0],
            n=[2, 2], c=[1e-6/BARSA, 1e-6/BARSA], p_ref=200*BARSA)
        W = add_well(None, g, rock, np.array([0]), well_type="rate",
                     val=1e-4, compi=np.array([1.0, 0.0]), sign=1, name="I")
        W = add_well(W, g, rock, np.array([g.cells.num - 1]), well_type="bhp",
                     val=150*BARSA, compi=np.array([0.0, 1.0]), sign=-1,
                     name="P")
        model = GenericBlackOilModelAD(g, rock, fluid, W)
        state = State(pressure=np.full(g.cells.num, 200*BARSA),
                      sW=np.full(g.cells.num, 0.2),
                      qWs=np.array([1e-4, 0.0]), qOs=np.array([0.0, 0.0]),
                      bhp=np.array([210*BARSA, 150*BARSA]))
        schedule = simple_schedule(np.full(5, 10.0*DAY), W=W)
        return model, state, schedule

    def test_the_hook_returns_mrsts_five_values():
        """The simulator unpacks exactly five; a shorter tuple raises inside
        the run, after the step has already been taken."""
        model, state, schedule = _case()
        fn = get_plot_after_step_ad(state, model, schedule)
        out = fn(model, [state], [], NonLinearSolver(), schedule, [0.0])
        assert len(out) == 5 and out[-1] is True
        print("ok test_the_hook_returns_mrsts_five_values")

    def test_it_is_called_once_per_report_step():
        model, state, schedule = _case()
        seen = []
        fn = get_plot_after_step_ad(
            state, model, schedule,
            plot_reservoir_fn=lambda G, states, current: seen.append(current))
        solver = NonLinearSolver()
        solver.verbose = False
        states, _ = solver.simulate_schedule(model, state, schedule,
                                             after_step_fn=fn)
        # One seeding call before the run, then one per accepted step.
        assert seen == [0, 0, 1, 2, 3, 4], seen
        assert len(states) == 5
        print("ok test_it_is_called_once_per_report_step")

    def test_it_is_handed_every_state_so_far_not_just_the_latest():
        """A monitor that sees one state can plot a point; MRST's plots a
        curve, and a stopping rule needs the trend."""
        model, state, schedule = _case()
        counts = []
        fn = get_plot_after_step_ad(
            state, model, schedule,
            plot_reservoir_fn=lambda G, states, current: counts.append(len(states)))
        solver = NonLinearSolver()
        solver.verbose = False
        solver.simulate_schedule(model, state, schedule, after_step_fn=fn)
        assert counts == [2, 1, 2, 3, 4, 5], counts
        print("ok test_it_is_handed_every_state_so_far_not_just_the_latest")

    def test_returning_not_ok_stops_the_run():
        """The veto, and the reason this is a hook rather than an observer."""
        model, state, schedule = _case()
        fn = get_plot_after_step_ad(
            state, model, schedule,
            stop_when=lambda model, states, reports: len(states) >= 2)
        solver = NonLinearSolver()
        solver.verbose = False
        states, report = solver.simulate_schedule(model, state, schedule,
                                                  after_step_fn=fn)
        assert len(states) == 2, len(states)
        assert len(report.steps) == 2
        print("ok test_returning_not_ok_stops_the_run")

    def test_no_hook_runs_the_whole_schedule():
        """The path every other caller takes must be untouched."""
        model, state, schedule = _case()
        solver = NonLinearSolver()
        solver.verbose = False
        states, _ = solver.simulate_schedule(model, state, schedule)
        assert len(states) == 5
        print("ok test_no_hook_runs_the_whole_schedule")

    def test_the_well_renderer_receives_well_solutions_and_times():
        model, state, schedule = _case()
        seen = {}

        def plot_well_fn(well_sols, time):
            seen["n"] = len(well_sols)
            seen["t"] = np.asarray(time, dtype=float)

        fn = get_plot_after_step_ad(state, model, schedule, plot_well_fn=plot_well_fn)
        solver = NonLinearSolver()
        solver.verbose = False
        solver.simulate_schedule(model, state, schedule, after_step_fn=fn)
        assert seen["n"] == 5, seen["n"]
        assert np.isclose(seen["t"][-1], 50.0*DAY), seen["t"]
        print("ok test_the_well_renderer_receives_well_solutions_and_times")

    def test_the_simulated_time_is_cumulative_not_per_step():
        """MRST hands `simtime`, the elapsed time at each report boundary.
        Per-step lengths would make a time axis that never advances."""
        model, state, schedule = _case()
        seen = []
        fn = get_plot_after_step_ad(state, model, schedule,
                                    plot_well_fn=lambda ws, t: seen.append(np.max(t)))
        solver = NonLinearSolver()
        solver.verbose = False
        solver.simulate_schedule(model, state, schedule, after_step_fn=fn)
        assert np.allclose(seen[1:], np.arange(1, 6)*10.0*DAY), seen
        print("ok test_the_simulated_time_is_cumulative_not_per_step")

    def _spe1_run(n_steps, **options):
        import matplotlib
        matplotlib.use("Agg")
        from pymrst.modules.ad_core.ad_oo.simulate_schedule_ad import simulate_schedule_ad
        from pymrst.modules.ad_core.utils.convert_deck_schedule_to_mrst import (
            convert_deck_schedule_to_mrst,
        )
        from pymrst.modules.ad_core.utils.schedule import Schedule
        from pymrst.modules.ad_core.utils.select_model_from_deck import select_model_from_deck
        from pymrst.modules.test_suite.dataset_functions.spe1 import setup_spe1
        from pymrst.params.gravity.gravity import isolated_gravity
        with isolated_gravity():
            G, rock, fluid, deck, state0 = setup_spe1()
            model = select_model_from_deck(G, rock, fluid, deck)
            full = convert_deck_schedule_to_mrst(model, deck)
            schedule = Schedule(step_val=full.step_val[:n_steps],
                                step_control=full.step_control[:n_steps],
                                controls=full.controls)
            fn = get_plot_after_step(state0, model, schedule, **options)
            ws, states, report = simulate_schedule_ad(state0, model, schedule, afterStepFn=fn)
        return model, ws, states, report

    def test_the_port_runs_the_schedule_with_its_runtime_panel():
        import matplotlib.pyplot as plt
        from pymrst.modules.ad_core.plotting.simulation_runtime_panel import PANEL_TAG
        model, ws, states, report = _spe1_run(2, plotWell=False, plotReservoir=False)
        assert len(states) == 2 and all(s is not None for s in states)
        assert plt.fignum_exists(PANEL_TAG)
        plt.close("all")
        print("ok test_the_port_runs_the_schedule_with_its_runtime_panel")

    def test_the_port_feeds_the_well_viewer():
        """Seeded with {ws; ws}, then fed every computed step's wellSol."""
        import matplotlib.pyplot as plt
        _, ws, states, _ = _spe1_run(2, plotWell=True, plotReservoir=False)
        names = [f.get_label() for f in map(plt.figure, plt.get_fignums())]
        assert "mrst-simpanel" in names and len(names) == 2, names
        plt.close("all")
        print("ok test_the_port_feeds_the_well_viewer")

    def test_the_converged_ministep_times_accumulate():
        reports = [{"StepReports": [{"Converged": True, "Timestep": 2.0},
                                    {"Converged": False, "Timestep": 9.0},
                                    {"Converged": True, "Timestep": 3.0}]},
                   {"StepReports": [{"Converged": True, "Timestep": 5.0}]}]
        np.testing.assert_array_equal(_get_timesteps(reports), [2.0, 5.0, 10.0])
        print("ok test_the_converged_ministep_times_accumulate")

    test_the_port_runs_the_schedule_with_its_runtime_panel()
    test_the_port_feeds_the_well_viewer()
    test_the_converged_ministep_times_accumulate()
    test_the_hook_returns_mrsts_five_values()
    test_it_is_called_once_per_report_step()
    test_it_is_handed_every_state_so_far_not_just_the_latest()
    test_returning_not_ok_stops_the_run()
    test_no_hook_runs_the_whole_schedule()
    test_the_well_renderer_receives_well_solutions_and_times()
    test_the_simulated_time_is_cumulative_not_per_step()
    print("\nAll tests passed!")
