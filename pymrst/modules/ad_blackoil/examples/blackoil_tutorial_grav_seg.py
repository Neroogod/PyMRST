"""MRST ``ad-blackoil/examples/blackoilTutorialGravSeg.m``.

Gravity segregation with the two-phase AD solvers: a 5x5x10 column of
1x1x5 m cells, water (1 cP, 1000 kg/m^3) above oil (10 cP, 700 kg/m^3),
linear relative permeabilities, gravity on. The top face is held at 100 bar
and lets oil in (`pside`, 'ZMin', saturation [0 1]); twenty 90-day steps of
`simulateScheduleAD` on a `TwoPhaseOilWaterModel` let the water sink and the
oil rise. `blackoil_tutorial_grav_seg` plots the initial water saturation,
runs the schedule verbosely, and animates the segregation beside the
initial state; it returns the model, schedule, initial state and results.

`build_blackoil_tutorial_grav_seg`/`run_blackoil_tutorial_grav_seg`,
below, are PyMRST's earlier version on the legacy models, kept for their
callers.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from pymrst.gridprocessing import cart_grid, compute_geometry
from pymrst.modules.ad_blackoil.models import GenericBlackOilModelAD
from pymrst.modules.ad_blackoil.examples.utils.schedule import ExampleResult, run_schedule
from pymrst.modules.ad_core import NonLinearSolver, State
from pymrst.modules.ad_props import init_simple_adi_fluid
from pymrst.params import make_rock, pside
from pymrst.utils.units import BARSA, CENTI, DARCY, DAY, MILLI, POISE


def blackoil_tutorial_grav_seg(plot: bool = True) -> SimpleNamespace:
    """MRST `blackoilTutorialGravSeg`."""
    from pymrst.modules.ad_blackoil.examples.utils.figures import (
        animate_cell_data, cell_data_figure,
    )
    from pymrst.modules.ad_blackoil.models import TwoPhaseOilWaterModel
    from pymrst.modules.ad_core.ad_oo.simulate_schedule_ad import simulate_schedule_ad
    from pymrst.modules.ad_core.utils.schedule import simple_schedule
    from pymrst.params.gravity.gravity import gravity
    from pymrst.solvers.init_res_sol import init_res_sol
    from pymrst.utils.format_time_range import format_time_range
    from pymrst.utils.gridtools.grid_logical_indices import grid_logical_indices

    # Reservoir geometry and petrophysical properties
    dims = [5, 5, 10]
    G = compute_geometry(cart_grid(dims, [1.0, 1.0, 5.0]))
    rock = make_rock(G, 100*MILLI*DARCY, 0.5)
    # Defining the fluid properties
    fluid = init_simple_adi_fluid(phases="WO", mu=np.array([1.0, 10.0])*CENTI*POISE,
                                  n=[1.0, 1.0], rho=[1000.0, 700.0])
    # Defining the model
    gravity("reset", "on")
    model = TwoPhaseOilWaterModel(G, rock, fluid)
    print(model)
    # Set up initial state: water above oil (layers 6-10 are oil)
    ii, jj, kk = grid_logical_indices(G)
    lower = kk + 1 > 5
    sW = np.ones(G.cells.num)
    sW[lower] = 0
    s = np.column_stack([sW, 1 - sW])
    state = init_res_sol(G, 100*BARSA, s)
    if plot:
        cell_data_figure(G, np.asarray(state.s)[:, 0], "Initial water saturation",
                         view=(50, 20))
    # Set up boundary conditions and timesteps
    bc = pside(None, G, "ZMin", 100*BARSA, sat=np.array([0.0, 1.0]))
    dt = np.full(20, 90*DAY)
    schedule = simple_schedule(dt, bc=bc)
    # Simulate the problem
    wellSols, states, report = simulate_schedule_ad(state, model, schedule, Verbose=True)
    # Plot the segregation process
    if plot:
        titles = [f"t={format_time_range(dt[:i + 1].sum())}" for i in range(len(states))]
        animate_cell_data(G, [np.asarray(st["s"])[:, 0] for st in states], titles,
                          view=(50, 20), clim=[0, 1], initial=np.asarray(state.s)[:, 0],
                          initial_title="T=0")
    return SimpleNamespace(G=G, model=model, schedule=schedule, state0=state,
                           wellSols=wellSols, states=states, report=report)


def build_blackoil_tutorial_grav_seg():
    dims = [5, 5, 10]
    g = compute_geometry(cart_grid(dims, [1.0, 1.0, 5.0]))
    rock = make_rock(g, 100 * MILLI * DARCY, 0.5)
    fluid = init_simple_adi_fluid(
        mu=np.array([1.0, 10.0]) * CENTI * POISE,
        rho=[1000.0, 700.0],
        n=[1, 1],
    )
    model = GenericBlackOilModelAD(g, rock, fluid, [], gravity=9.80665,
                                 tol_cnv=1e-7, tol_mb=1e-11)
    k = np.arange(g.cells.num) // (dims[0] * dims[1])
    sw = np.ones(g.cells.num)
    sw[k > 4] = 0.0
    state = State(
        pressure=np.full(g.cells.num, 100 * BARSA),
        sW=sw,
        qWs=np.zeros(0),
        qOs=np.zeros(0),
        bhp=np.zeros(0),
    )
    bc = pside(None, g, "zmin", 100 * BARSA, sat=np.array([0.0, 1.0]))
    # MRST: `dt = repmat(90*day, 20, 1)`. Eight steps stop the segregation
    # partway; the example is about where it *settles*.
    timesteps = np.full(20, 90.0 * DAY)
    return model, state, timesteps, bc


def run_blackoil_tutorial_grav_seg() -> ExampleResult:
    model, state, timesteps, bc = build_blackoil_tutorial_grav_seg()
    solver = NonLinearSolver(max_iterations=30, max_timestep_cuts=8)
    return run_schedule(
        model, state, timesteps, controls={"bc": bc},
        history_keys=("pressure", "sW"), solver=solver,
    )


blackoilTutorialGravSeg = blackoil_tutorial_grav_seg

__all__ = ["blackoilTutorialGravSeg", "blackoil_tutorial_grav_seg",
           "build_blackoil_tutorial_grav_seg", "run_blackoil_tutorial_grav_seg"]


if __name__ == "__main__":
    import pyvista as pv

    from pymrst.params.gravity.gravity import isolated_gravity

    pv.OFF_SCREEN = True
    with isolated_gravity():
        OUT = blackoil_tutorial_grav_seg(plot=True)

    def test_the_water_sinks_below_the_oil():
        """Segregation ends with the heavier water at the bottom."""
        sw = np.asarray(OUT.states[-1]["s"])[:, 0].reshape(10, 25)
        assert sw[-1].mean() > 0.9 and sw[0].mean() < 0.1, sw.mean(axis=1)
        print("ok test_the_water_sinks_below_the_oil")

    def test_the_run_is_mrsts_twenty_90_day_steps():
        assert len(OUT.states) == 20
        np.testing.assert_allclose(OUT.schedule.step_val, 90*DAY)
        assert type(OUT.model).__name__ == "TwoPhaseOilWaterModel"
        print("ok test_the_run_is_mrsts_twenty_90_day_steps")

    test_the_water_sinks_below_the_oil()
    test_the_run_is_mrsts_twenty_90_day_steps()
    print("\nAll tests passed!")
