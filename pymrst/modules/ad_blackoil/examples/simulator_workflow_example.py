"""Port of MRST ``simulatorWorkflowExample.m``."""

from __future__ import annotations

import numpy as np

from pymrst.gridprocessing import cart_grid, compute_geometry
from pymrst.modules.ad_blackoil.models import GenericBlackOilModel
from pymrst.modules.ad_blackoil.examples.utils.schedule import ExampleResult, run_schedule
from pymrst.modules.ad_core import State
from pymrst.modules.ad_props import init_simple_adi_fluid
from pymrst.params import add_well, make_rock
from pymrst.utils.units import BARSA, CENTI, DARCY, DAY, MILLI, POISE


def build_simulator_workflow_example():
    g = compute_geometry(cart_grid([12, 12, 1], [300.0, 300.0, 10.0]))
    rock = make_rock(g, 100 * MILLI * DARCY, 0.25)
    fluid = init_simple_adi_fluid(
        mu=np.array([1.0, 5.0]) * CENTI * POISE,
        rho=[1000.0, 800.0],
        n=[2, 2],
        c=[1e-6 / BARSA, 1e-4 / BARSA],
        p_ref=200 * BARSA,
    )
    wells = add_well(None, g, rock, np.array([0]), well_type="rate",
                     val=1e-4, compi=np.array([1.0, 0.0]), sign=1, name="I")
    wells = add_well(wells, g, rock, np.array([g.cells.num - 1]),
                     well_type="bhp", val=150 * BARSA,
                     compi=np.array([0.0, 1.0]), sign=-1, name="P")
    model = GenericBlackOilModel(g, rock, fluid, wells)
    state = State(
        pressure=np.full(g.cells.num, 200 * BARSA),
        sW=np.full(g.cells.num, 0.2),
        qWs=np.array([1e-4, 0.0]),
        qOs=np.array([0.0, 0.0]),
        bhp=np.array([200 * BARSA, 150 * BARSA]),
    )
    timesteps = np.full(5, 30.0 * DAY)
    return model, state, timesteps


def run_simulator_workflow_example() -> ExampleResult:
    model, state, timesteps = build_simulator_workflow_example()
    return run_schedule(model, state, timesteps,
                        history_keys=("pressure", "sW", "qWs", "qOs", "bhp"))


if __name__ == "__main__":
    result = run_simulator_workflow_example()
    print(f"workflow completed in {result.report.total_iterations} Newton iterations")
