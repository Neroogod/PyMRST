"""MRST ``ad-core/examples/demoPackedProblems.m``.

Packed simulation problems: a 1D waterflood (100 cells over 1000 m, 0.1 darcy,
30 % porosity; water injected at one pore volume over ten years in the first
cell, production at 100 bar from the last, ramped-up steps of a hundredth of
the time) packed twice under the base name 'test_packed', once with linear and
once with quadratic relative permeabilities. Both are simulated
(`simulatePackedProblem`), their output read back one at a time and together,
and the producers' water rates plotted with `plotWellSols`. The output is then
cleared, first asking (`prompt`), then without asking, and both problems are
simulated again in background sessions while `monitorBackgroundSimulations`
shows their progress.

`plot` draws the well curves; `background=False` stops before the background
runs, which in PyMRST need `cloudpickle` to store the models.

`demo_packed_problems_ad` (in `packed_problem_examples.py`) is PyMRST's earlier
surrogate, kept for its callers.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from pymrst.gridprocessing import cart_grid, compute_geometry
from pymrst.modules.ad_blackoil.models.two_phase_oil_water_model import TwoPhaseOilWaterModel
from pymrst.modules.ad_core.simulators.sim_runner.clear_packed_simulator_output import (
    clear_packed_simulator_output,
)
from pymrst.modules.ad_core.simulators.sim_runner.get_multiple_packed_simulator_outputs import (
    get_multiple_packed_simulator_outputs,
)
from pymrst.modules.ad_core.simulators.sim_runner.get_packed_simulator_output import (
    get_packed_simulator_output,
)
from pymrst.modules.ad_core.simulators.sim_runner.monitor_background_simulations import (
    monitor_background_simulations,
)
from pymrst.modules.ad_core.simulators.sim_runner.pack_simulation_problem import (
    pack_simulation_problem,
)
from pymrst.modules.ad_core.simulators.sim_runner.simulate_packed_problem import (
    simulate_packed_problem,
)
from pymrst.modules.ad_core.simulators.sim_runner.simulate_packed_problem_background import (
    simulate_packed_problem_background,
)
from pymrst.modules.ad_core.timesteps.rampup_timesteps import rampup_timesteps
from pymrst.modules.ad_core.utils.simple_schedule import simple_schedule
from pymrst.modules.ad_props import init_simple_adi_fluid
from pymrst.params.rock import make_rock
from pymrst.params.wells_and_bc import add_well
from pymrst.solvers.init_res_sol import init_res_sol
from pymrst.utils.units import BARSA, CENTI, DARCY, POISE, YEAR

BASE_NAME = "test_packed"


def demo_packed_problems(plot: bool = False, prompt: bool = True,
                         background: bool = True) -> SimpleNamespace:
    """MRST `demoPackedProblems`: the problems and what was read back."""
    problems = pack_demo_problems()
    ok, status = simulate_packed_problem(problems)
    ws_1, states_1, reports_1 = get_packed_simulator_output(problems[0])[:3]
    ws_2, states_2, reports_2 = get_packed_simulator_output(problems[1])[:3]
    all_ws, all_states, all_reports, names, report_time = \
        get_multiple_packed_simulator_outputs(problems)[:5]
    if plot:
        from pymrst.modules.ad_core.plotting.plot_well_sols import plot_well_sols
        plot_well_sols(all_ws, report_time, DatasetNames=names, field="qWs",
                       SelectedWells=[2])
    out = SimpleNamespace(problems=problems, ok=ok, status=status,
                          ws_1=ws_1, states_1=states_1, reports_1=reports_1,
                          ws_2=ws_2, states_2=states_2, reports_2=reports_2,
                          all_ws=all_ws, all_states=all_states, all_reports=all_reports,
                          names=names, report_time=report_time)
    clear_packed_simulator_output(problems, Prompt=prompt)
    clear_packed_simulator_output(problems, Prompt=False)
    if background:
        for problem in problems:
            simulate_packed_problem_background(problem, verbose=True)
        monitor_background_simulations(problems)
    return out


def pack_demo_problems() -> list:
    """The two packed problems of the demo."""
    G = cart_grid([100, 1], [1000.0, 1000.0])
    rock = make_rock(G, 0.1*DARCY, 0.3)
    G = compute_geometry(G)
    fluid_1 = init_simple_adi_fluid(mu=[1*CENTI*POISE, 1*CENTI*POISE], n=[1, 1],
                                    phases="wo", rho=[1000, 500])
    model_1 = TwoPhaseOilWaterModel(G, rock, fluid_1)
    time = 10*YEAR
    irate = np.sum(model_1.operators.pv)/time
    W = add_well(None, G, rock, np.array([0]), well_type="rate", val=irate,
                 compi=np.array([1.0, 0.0]))
    W = add_well(W, G, rock, np.array([G.cells.num - 1]), well_type="bhp", val=100*BARSA,
                 compi=np.array([1.0, 0.0]))
    dt = rampup_timesteps(time, time/100)
    schedule = simple_schedule(dt, W=W)
    state0 = init_res_sol(G, 150*BARSA, np.array([0.0, 1.0]))
    fluid_2 = init_simple_adi_fluid(mu=[1*CENTI*POISE, 1*CENTI*POISE], n=[2, 2],
                                    phases="wo", rho=[1000, 500])
    model_2 = TwoPhaseOilWaterModel(G, rock, fluid_2)
    problem_1 = pack_simulation_problem(state0, model_1, schedule, BASE_NAME,
                                        Name="linear_relperm",
                                        Description="1D displacement with linear flux")
    problem_2 = pack_simulation_problem(state0, model_2, schedule, BASE_NAME,
                                        Name="quadratic_relperm",
                                        Description="1D displacement with nonlinear flux")
    return [problem_1, problem_2]


demoPackedProblems = demo_packed_problems

__all__ = ["demoPackedProblems", "demo_packed_problems", "pack_demo_problems"]


if __name__ == "__main__":
    import contextlib
    import io

    def test_both_problems_simulate_and_read_back():
        with contextlib.redirect_stdout(io.StringIO()):
            clear_packed_simulator_output(pack_demo_problems(), Prompt=False)
            out = demo_packed_problems(prompt=False, background=False)
        nsteps = out.problems[0].SimulatorSetup.schedule.step_val.size
        assert out.ok == [True, True]
        assert len(out.states_1) == len(out.states_2) == nsteps
        assert list(out.names) == ["linear_relperm", "quadratic_relperm"]
        assert all(p.OutputHandlers.states.numel_data() == 0 for p in out.problems)
        q1 = out.ws_1[-1][1].qWs
        q2 = out.ws_2[-1][1].qWs
        assert q1 != q2
        print("ok test_both_problems_simulate_and_read_back")

    test_both_problems_simulate_and_read_back()
    print("\nAll tests passed!")
