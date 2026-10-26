"""MRST ``ad-blackoil/examples/immiscibleTimeIntegrationExample.m``.

Time integration of a 1D waterflood: 150 cells over 1000 m, porosity 0.5
with three low-porosity bands (0.05, 0.1, 0.05), water (1 Pa s, linear
relative permeabilities) injected at the left at one pore volume over 500
days against 0 bar at the right. A `GenericBlackOilModel` is solved over
ramped-up steps of 500/150 days with four time discretizations
(`setTimeDiscretization`):
- fully implicit;
- explicit, its steps limited to saturation and composition CFL 0.9;
- adaptive implicit (AIM), implicit only where the CFL exceeds 0.9;
- explicit with CFL 5 and `stepFunctionIsLinear` - unstable, by design.
They are packed (`packSimulationProblem`, base name 'immiscible_time') and
simulated, and the saturations are compared step by step, with porosity and
the fully implicit solution's saturation CFL; AIM's fraction of implicit
cells and each run's ministeps are plotted. Finally the model is made
sequential (`getSequentialModelFromFI`) and run sequential-implicit,
sequential-explicit and "sequential-AIM" alongside the first three (MRST's
"AIM" sequential variant is set up with 'explicit', so it repeats the
explicit one; that is kept).

MRST warns that the example takes long and pauses ten seconds; the port does
so when `plot` is on. `build_immiscible_time_integration_example`/
`run_immiscible_time_integration_example`, below, are PyMRST's earlier
fully implicit version, kept for their callers.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from pymrst.gridprocessing import cart_grid, compute_geometry
from pymrst.modules.ad_blackoil.models import GenericBlackOilModelAD
from pymrst.modules.ad_blackoil.examples.utils.schedule import ExampleResult, run_schedule
from pymrst.modules.ad_core import State, rampup_timesteps
from pymrst.modules.ad_props import init_simple_adi_fluid
from pymrst.params import fluxside, make_rock, pore_volume, pside
from pymrst.utils.units import BARSA, DARCY, DAY


REGIONS = ([0.20, 0.30, 0.05], [0.45, 0.55, 0.1], [0.70, 0.80, 0.05])


def immiscible_time_integration_example(plot: bool = True, sequential: bool = True) -> SimpleNamespace:
    """MRST `immiscibleTimeIntegrationExample`. `sequential=False` stops
    before the sequential section."""
    import time as _time
    import warnings

    from pymrst.modules.ad_blackoil.models import GenericBlackOilModel
    from pymrst.modules.ad_core.simulators.sim_runner.get_multiple_packed_simulator_outputs import (
        get_multiple_packed_simulator_outputs,
    )
    from pymrst.modules.ad_core.simulators.sim_runner.pack_simulation_problem import (
        pack_simulation_problem,
    )
    from pymrst.modules.ad_core.simulators.sim_runner.simulate_packed_problem import (
        simulate_packed_problem,
    )
    from pymrst.modules.ad_core.utils.get_report_ministeps import get_report_ministeps
    from pymrst.modules.ad_core.utils.schedule import simple_schedule
    from pymrst.modules.ad_core.utils.set_time_discretization import set_time_discretization
    from pymrst.solvers.init_res_sol import init_res_sol

    # Notice on Computational Cost
    warnings.warn("Please be advised that this example often takes a long time to run "
                  "(e.g., more than 30 minutes of CPU time)")
    if plot:
        _time.sleep(10)
    L = 1000.0
    G = compute_geometry(cart_grid([150], [L]))
    x = np.asarray(G.cells.centroids, dtype=float).ravel()/L
    poro = np.full(G.cells.num, 0.5)
    for R in REGIONS:
        poro[(x > R[0]) & (x < R[1])] = R[2]
    rock = make_rock(G, 1*DARCY, poro)
    if plot:
        import matplotlib.pyplot as plt
        plt.close("all")
        plt.plot(poro)
    time = 500*DAY
    fluid = init_simple_adi_fluid(phases="WO", n=[1, 1])
    model = GenericBlackOilModel(G, rock, fluid, gas=False)
    state0 = init_res_sol(G, 50*BARSA, np.array([0.0, 1.0]))
    pv = pore_volume(G, rock)
    injRate = np.sum(pv)/time
    bc = fluxside(None, G, "xmin", injRate, sat=np.array([1.0, 0.0]))
    bc = pside(bc, G, "xmax", 0*BARSA, sat=np.array([0.0, 1.0]))
    n = 150
    dT = time/n
    timesteps = rampup_timesteps(time, dT, 8)
    schedule = simple_schedule(timesteps, bc=bc)
    # Fully-implicit
    implicit = pack_simulation_problem(state0, model, schedule, "immiscible_time",
                                       Name="FIM", Description="Fully-implicit")
    # Explicit solver
    model_explicit = set_time_discretization(model, "explicit", verbose=2)
    explicit = pack_simulation_problem(state0, model_explicit, schedule, "immiscible_time",
                                       Name="EXPL", Description="Explicit")
    fsb = model_explicit.FlowDiscretization.get_flow_state_builder()
    print(fsb)
    # Adaptive implicit
    model_aim = set_time_discretization(model, "adaptive-implicit", verbose=2)
    aim = pack_simulation_problem(state0, model_aim, schedule, "immiscible_time",
                                  Name="AIM", Description="AIM")
    # Make an explicit solver with larger CFL limit
    model_explicit_largedt = set_time_discretization(model, "explicit", verbose=True,
                                                     saturationCFL=5,
                                                     compositionCFL=np.inf)  # Immiscible
    model_explicit_largedt.stepFunctionIsLinear = True
    explicit_largedt = pack_simulation_problem(state0, model_explicit_largedt, schedule,
                                               "immiscible_time",
                                               Description="Explicit (CFL>1)",
                                               Name="EXPL_unstable")
    # Simulate the problems
    problems = [implicit, explicit, aim, explicit_largedt]
    simulate_packed_problem(problems, continueOnError=False)
    # Get output and plot the well results
    ws, states, reports, *_ = get_multiple_packed_simulator_outputs(problems)
    names = [p.Description for p in problems]
    model_cfl = model.setup_state_function_groupings()
    cfl = _cfl_per_step(model_cfl, states[0], schedule)
    # Plot the fraction of cells which are implicit
    imp_frac = np.array([np.sum(x["implicit"])/np.size(x["implicit"]) for x in states[2]])
    ministeps = [get_report_ministeps(r) for r in reports]
    out = SimpleNamespace(G=G, rock=rock, model=model, schedule=schedule, state0=state0,
                          problems=problems, names=names, ws=ws, states=states,
                          reports=reports, cfl=cfl, imp_frac=imp_frac, ministeps=ministeps)
    if plot:
        _plot_runs(out)
    # Solve with sequential version of the solver
    if sequential:
        from pymrst.modules.sequential.utils.get_sequential_model_from_fi import (
            get_sequential_model_from_fi,
        )
        modelseq = get_sequential_model_from_fi(model)
        seq = pack_simulation_problem(state0, modelseq, schedule, "immiscible_time",
                                      Description="Sequential-Implicit", Name="SI")
        modelseq_explicit = set_time_discretization(modelseq, "explicit", verbose=True)
        seq_explicit = pack_simulation_problem(state0, modelseq_explicit, schedule,
                                               "immiscible_time",
                                               Description="Sequential-Explicit", Name="SE")
        modelseq_aim = set_time_discretization(modelseq, "explicit", verbose=True)
        seq_aim = pack_simulation_problem(state0, modelseq_aim, schedule, "immiscible_time",
                                          Description="Sequential-AIM", Name="SAIM")
        allvariants = [implicit, explicit, aim, seq, seq_explicit, seq_aim]
        simulate_packed_problem(allvariants, continueOnError=False)
        ws_all, states_all, reports_all, snames, T, *_ = get_multiple_packed_simulator_outputs(
            allvariants)
        # MRST takes the names of `problems` again here, so only four.
        out.allvariants = allvariants
        out.ws_all, out.states_all, out.reports_all, out.T = ws_all, states_all, reports_all, T
        if plot:
            _plot_displacement(states_all, T, names)
    return out


immiscibleTimeIntegrationExample = immiscible_time_integration_example


def _cfl_per_step(model_cfl, states, schedule) -> np.ndarray:
    """`estimateSaturationCFL(model_cfl, state, dt)` for each report step."""
    from pymrst.modules.ad_core.utils.estimate_saturation_cfl import estimate_saturation_cfl
    return np.array([estimate_saturation_cfl(model_cfl, st, dt)
                     for st, dt in zip(states, schedule.step_val)])


def _plot_cells(ax, G, data, **options):
    """MATLAB's `plotCellData` on a 1D grid: the values at the cell centres."""
    return ax.plot(np.asarray(G.cells.centroids, dtype=float).ravel(), np.asarray(data).ravel(),
                   **options)[0]


def _plot_runs(out: SimpleNamespace) -> None:
    import matplotlib.pyplot as plt
    G, states, names = out.G, out.states, out.names
    fig = plt.figure(1)
    for step in range(len(out.schedule.step_val)):
        fig.clf()
        ax = fig.add_subplot(2, 1, 1)
        for run in states:
            _plot_cells(ax, G, np.asarray(run[step]["s"])[:, 0])
        ax.legend(names)
        ax = fig.add_subplot(2, 1, 2)
        _plot_cells(ax, G, out.rock.poro)
        _plot_cells(ax, G, out.cfl[step])
        _plot_cells(ax, G, np.ones(G.cells.num), color="k")
        ax.legend(["Porosity", "CFL", "CFL stable limit"])
        plt.pause(0.001)
    L = 1000.0
    for step in range(len(out.schedule.step_val)):
        fig.clf()
        ax = fig.add_subplot(1, 1, 1)
        for R in REGIONS:
            x1, x2 = L*R[0], L*R[1]
            ax.fill([x1, x1, x2, x2], [0, 1, 1, 0], "k", alpha=.1)
            ax.text((x1 + x2)/2, 0.8, f"$\\phi$ = {R[2]:1.2f}", ha="center", fontsize=14)
        h = [_plot_cells(ax, G, np.asarray(run[step]["s"])[:, 0],
                         linestyle="--" if j == 0 else "-", linewidth=2 if j == 0 else 1)
             for j, run in enumerate(states)]
        ax.legend(h, names, fontsize=13)
        plt.pause(0.001)
    plt.figure()
    plt.plot(np.arange(1, out.imp_frac.size + 1), 100*out.imp_frac)
    plt.title("Number of implicit cells in AIM")
    plt.ylabel("Percentage of implicit cells")
    plt.xlabel("Step index")
    plt.figure()
    for dt, marker in zip(out.ministeps, ("o", ".", "x", ".")):
        frac = np.arange(1, dt.size + 1)/dt.size
        plt.plot(frac, dt/DAY, marker, linestyle="none")
    plt.ylabel("Timestep length [day]")
    plt.xlabel("Fraction of total simulation time")
    plt.legend(names)
    plt.show()


def _plot_displacement(states, T, names) -> None:
    import matplotlib.pyplot as plt
    markers = ("-.", ".", "o")
    fig = plt.figure()
    for i in range(len(T[0])):
        fig.clf()
        ax = fig.add_subplot(1, 1, 1)
        for j, run in enumerate(states):
            ax.plot(np.asarray(run[i]["s"])[:, 0], markers[(j + 1) % 3])
        ax.legend(names)
        plt.pause(0.001)
    plt.show()


def build_immiscible_time_integration_example(ncells: int = 150):
    """MRST `immiscibleTimeIntegrationExample.m`, parameter for parameter.

    MRST uses `cartGrid(150, L)` and `dT = time/150`; this port had 75 cells
    and `time/50`, which changes both the resolution the fronts are seen at
    and the CFL number every explicit step is judged against - the two things
    the example exists to compare.
    """
    length = 1000.0
    g = compute_geometry(cart_grid([ncells, 1, 1], [length, 1.0, 1.0]))
    x = g.cells.centroids[:, 0] / length
    poro = np.full(g.cells.num, 0.5)
    for lo, hi, val in [(0.20, 0.30, 0.05), (0.45, 0.55, 0.1), (0.70, 0.80, 0.05)]:
        poro[(x > lo) & (x < hi)] = val
    rock = make_rock(g, 1.0 * DARCY, poro[:, None])
    fluid = init_simple_adi_fluid(mu=[1e-3, 1e-3], rho=[1000.0, 800.0], n=[1, 1])
    model = GenericBlackOilModelAD(g, rock, fluid, [], tol_cnv=1e-8, tol_mb=1e-12)
    state = State(
        pressure=np.full(g.cells.num, 50 * BARSA),
        sW=np.zeros(g.cells.num),
        qWs=np.zeros(0),
        qOs=np.zeros(0),
        bhp=np.zeros(0),
    )
    total_time = 500.0 * DAY
    inj_rate = float(pore_volume(g, rock).sum() / total_time)
    bc = fluxside(None, g, "xmin", inj_rate, sat=np.array([1.0, 0.0]))
    bc = pside(bc, g, "xmax", 0.0, sat=np.array([0.0, 1.0]))
    # MRST: `dT = time/n` with `n = 150` - the same count as the grid, which
    # is not a coincidence: one step per cell puts the CFL number near one for
    # the explicit scheme, which is the regime being demonstrated.
    timesteps = rampup_timesteps(total_time, total_time / float(ncells), 8)
    return model, state, timesteps, bc, poro


def estimate_saturation_cfl_ad(result: ExampleResult, timesteps) -> np.ndarray:
    sw = result.history["sW"]
    if sw.shape[0] < 2:
        return np.zeros_like(sw)
    dsw = np.abs(np.diff(sw, axis=0))
    dt = np.asarray(timesteps, dtype=float)[1:, None]
    return dsw / np.maximum(dt / DAY, 1e-300)


def run_immiscible_time_integration_example() -> tuple[ExampleResult, np.ndarray]:
    model, state, timesteps, bc, _ = build_immiscible_time_integration_example()
    result = run_schedule(
        model, state, timesteps, controls={"bc": bc},
        history_keys=("pressure", "sW"),
    )
    return result, estimate_saturation_cfl_ad(result, timesteps)



__all__ = ["build_immiscible_time_integration_example", "estimate_saturation_cfl_ad",
           "immiscibleTimeIntegrationExample", "immiscible_time_integration_example",
           "run_immiscible_time_integration_example"]


if __name__ == "__main__":
    import tempfile

    from pymrst.utils.settings_manager.mrst_settings import override_setting

    with tempfile.TemporaryDirectory() as tmp:
        with override_setting("outputDirectory", tmp):
            OUT = immiscible_time_integration_example(plot=False, sequential=False)

    def test_the_four_runs_cover_the_schedule():
        assert OUT.names == ["Fully-implicit", "Explicit", "AIM", "Explicit (CFL>1)"]
        assert all(len(run) == 158 for run in OUT.states)
        for dt in OUT.ministeps:
            np.testing.assert_allclose(dt.sum(), 500*DAY, rtol=1e-12)
        print("ok test_the_four_runs_cover_the_schedule")

    def test_the_explicit_run_keeps_its_cfl():
        """CFL 0.9 needs many more, shorter steps than the report steps."""
        assert OUT.ministeps[1].size > 5*OUT.ministeps[0].size
        assert OUT.ministeps[0].size == 158
        print("ok test_the_explicit_run_keeps_its_cfl")

    def test_aim_is_implicit_only_where_needed():
        assert OUT.imp_frac[0] == 1.0, "the first step has no fluxes: all implicit"
        assert 0 < OUT.imp_frac[1:].mean() < 1
        print("ok test_aim_is_implicit_only_where_needed")

    def test_the_stable_runs_agree_on_the_front():
        """Fully implicit, explicit and AIM put the front in the same place."""
        sw = [np.asarray(run[60]["s"])[:, 0] for run in OUT.states[:3]]
        front = [np.argmax(s < 0.5) for s in sw]
        assert max(front) - min(front) <= 3, front
        print("ok test_the_stable_runs_agree_on_the_front")

    test_the_four_runs_cover_the_schedule()
    test_the_explicit_run_keeps_its_cfl()
    test_aim_is_implicit_only_where_needed()
    test_the_stable_runs_agree_on_the_front()
    print("\nAll tests passed!")
