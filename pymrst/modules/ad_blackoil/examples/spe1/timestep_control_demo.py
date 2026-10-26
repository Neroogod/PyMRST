"""MRST ``ad-blackoil/examples/spe1/timestepControlDemo.m``.

Automated time-step control on SPE1. The schedule is first run as given.
Its report steps are then merged into one step per control period
(`compressSchedule`). The merged schedule is run four times, each with an
`IterationCountTimeStepSelector` aiming at 4, 8, 15 and 25 Newton
iterations per step, with a one-day first step, step changes of at most a
factor 4, and at most four times the target iterations per step. The
ministeps each run actually took (`convertReportToSchedule`) are plotted
against time; the well curves of all five runs go to `plotWellSols`; and
the total Newton iterations and wall time per run are compared in a bar
chart.

`timestep_control_demo` returns the runs and those quantities. Without
`plot` nothing is drawn.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np

TARGET_ITS = (4, 8, 15, 25)


def timestep_control_demo(plot: bool = True) -> SimpleNamespace:
    """MRST `timestepControlDemo`."""
    from pymrst.modules.ad_core.ad_oo.nonlinear_solver import NonLinearSolver
    from pymrst.modules.ad_core.ad_oo.simulate_schedule_ad import simulate_schedule_ad
    from pymrst.modules.ad_core.timesteps.iteration_count_time_step_selector import (
        IterationCountTimeStepSelector,
    )
    from pymrst.modules.ad_core.utils.convert_deck_schedule_to_mrst import (
        convert_deck_schedule_to_mrst,
    )
    from pymrst.modules.ad_core.utils.convert_report_to_schedule import (
        convert_report_to_schedule,
    )
    from pymrst.modules.ad_core.utils.schedule import compress_schedule
    from pymrst.modules.ad_core.utils.select_model_from_deck import select_model_from_deck
    from pymrst.modules.test_suite.dataset_functions.spe1 import setup_spe1
    from pymrst.utils.units import DAY

    # Load and run the SPE1 model
    G, rock, fluid, deck, state = setup_spe1()
    model = select_model_from_deck(G, rock, fluid, deck)
    print(model)
    schedule = convert_deck_schedule_to_mrst(model, deck)
    wellSols, states, report = simulate_schedule_ad(state, model, schedule)

    # Time-step control with iteration targets
    schedule_small = compress_schedule(schedule)
    rampup = 1*DAY
    reports, ws = [], []
    for target in TARGET_ITS:
        # Set up time selection class
        timestepper = IterationCountTimeStepSelector(
            target_iteration_count=target, min_relative_adjustment=np.sqrt(np.finfo(float).eps),
            max_relative_adjustment=4, first_rampup_step=rampup, verbose=True)
        # Instantiate a nonlinear solver with the timestep class as a
        # construction argument.
        nonlinear = NonLinearSolver(timeStepSelector=timestepper, maxiterations=4*target)
        # Solve and store results.
        ws_i, _, report_i = simulate_schedule_ad(state, model, schedule_small,
                                                 NonLinearSolver=nonlinear,
                                                 OutputMinisteps=True)
        ws.append(ws_i)
        reports.append(report_i)

    # Combine output
    labels = ["Base case"] + [f"Target: {t} its" for t in TARGET_ITS]
    wsols = [wellSols] + ws
    timesteps = [np.asarray(schedule.step_val, dtype=float)]
    for report_i in reports:
        # Read out the actually used timesteps
        _, t, _ = convert_report_to_schedule(report_i, schedule_small)
        timesteps.append(np.asarray(t, dtype=float))
    time = [np.cumsum(t) for t in timesteps]

    # Compare computational efficiency
    reps = [report] + reports
    iterations = np.zeros(len(reps))
    simulation_time = np.zeros(len(reps))
    for i, r in enumerate(reps):
        simulation_time[i] = np.sum(r["SimulationTime"])
        # MRST loops `for j = 1:numel(reps{i})`, once for a report struct;
        # its `if r.Converged` on the vector of step flags holds when all
        # of them are true.
        for k in range(len(r["ControlstepReports"])):
            if np.all(r["Converged"]):
                iterations[i] += r["ControlstepReports"][k]["Iterations"]
    out = SimpleNamespace(model=model, schedule=schedule, schedule_small=schedule_small,
                          wsols=wsols, reports=reps, labels=labels, timesteps=timesteps,
                          time=time, iterations=iterations, simulation_time=simulation_time)
    if plot:
        _plot(out)
    return out


timestepControlDemo = timestep_control_demo


def _plot(out: SimpleNamespace) -> None:
    import matplotlib.pyplot as plt
    from pymrst.modules.ad_core.plotting.plot_well_sols import plot_well_sols
    from pymrst.utils.units import DAY
    plt.figure()
    colours = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    for i, (t, dt) in enumerate(zip(out.time, out.timesteps)):
        plt.plot(t/DAY, dt/DAY, "-o", linewidth=2, color=colours[i % len(colours)])
        plt.grid(True)
    plt.legend(out.labels, loc="upper left")
    # Compare solution accuracy (well curves)
    plot_well_sols(out.wsols, out.time, datasetnames=out.labels)
    # Compare computational efficiency
    fig, ax = plt.subplots()
    x = np.arange(len(out.labels))
    ax.bar(x - 0.2, out.iterations, width=0.4)
    ax.bar(x + 0.2, out.simulation_time, width=0.4)
    ax.legend(["Non-linear iterations", "Time taken"])
    ax.set_xticks(x, out.labels)
    plt.show()


__all__ = ["timestepControlDemo", "timestep_control_demo"]


if __name__ == "__main__":
    from pymrst.params.gravity.gravity import isolated_gravity

    with isolated_gravity():
        OUT = timestep_control_demo(plot=False)

    def test_each_run_covers_the_whole_schedule():
        total = np.sum(OUT.schedule.step_val)
        for t in OUT.time:
            assert np.isclose(t[-1], total), (t[-1], total)
        print("ok test_each_run_covers_the_whole_schedule")

    def test_the_schedule_is_compressed_to_its_control_periods():
        assert OUT.schedule_small.step_val.size == np.unique(OUT.schedule.step_control).size
        print("ok test_the_schedule_is_compressed_to_its_control_periods")

    def test_a_higher_target_takes_fewer_longer_steps():
        counts = [t.size for t in OUT.timesteps[1:]]
        assert counts == sorted(counts, reverse=True), counts
        print("ok test_a_higher_target_takes_fewer_longer_steps")

    def test_ministep_well_solutions_match_the_ministeps():
        for ws, t in zip(OUT.wsols[1:], OUT.timesteps[1:]):
            assert len(ws) == t.size
        print("ok test_ministep_well_solutions_match_the_ministeps")

    test_each_run_covers_the_whole_schedule()
    test_the_schedule_is_compressed_to_its_control_periods()
    test_a_higher_target_takes_fewer_longer_steps()
    test_ministep_well_solutions_match_the_ministeps()
    print("\nAll tests passed!")
