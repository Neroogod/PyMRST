"""Port of MRST ``blackoilTutorialGravSeg.m``."""

from __future__ import annotations

import numpy as np

from pymrst.gridprocessing import cart_grid, compute_geometry
from pymrst.modules.ad_blackoil.models import GenericBlackOilModel
from pymrst.modules.ad_blackoil.examples.utils.schedule import ExampleResult, run_schedule
from pymrst.modules.ad_core import NonLinearSolver, State
from pymrst.modules.ad_props import init_simple_adi_fluid
from pymrst.params import make_rock, pside
from pymrst.utils.units import BARSA, CENTI, DARCY, DAY, MILLI, POISE


def build_blackoil_tutorial_grav_seg():
    dims = [5, 5, 10]
    g = compute_geometry(cart_grid(dims, [1.0, 1.0, 5.0]))
    rock = make_rock(g, 100 * MILLI * DARCY, 0.5)
    fluid = init_simple_adi_fluid(
        mu=np.array([1.0, 10.0]) * CENTI * POISE,
        rho=[1000.0, 700.0],
        n=[1, 1],
    )
    model = GenericBlackOilModel(g, rock, fluid, [], gravity=9.80665,
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
    timesteps = np.full(8, 90.0 * DAY)
    return model, state, timesteps, bc


def run_blackoil_tutorial_grav_seg() -> ExampleResult:
    model, state, timesteps, bc = build_blackoil_tutorial_grav_seg()
    solver = NonLinearSolver(max_iterations=30, max_timestep_cuts=8)
    return run_schedule(
        model, state, timesteps, controls={"bc": bc},
        history_keys=("pressure", "sW"), solver=solver,
    )


if __name__ == "__main__":
    result = run_blackoil_tutorial_grav_seg()
    sw = result.history["sW"][-1]
    print(f"final water saturation range {sw.min():.3f}-{sw.max():.3f}")
