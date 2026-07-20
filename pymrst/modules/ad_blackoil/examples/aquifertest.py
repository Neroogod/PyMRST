"""Port of MRST ``aquifertest.m`` using PyMRST's analytic aquifer model."""

from __future__ import annotations

import numpy as np

from pymrst.gridprocessing import cart_grid, compute_geometry
from pymrst.modules.ad_blackoil.models import GenericBlackOilModel
from pymrst.modules.ad_blackoil.examples.utils.schedule import ExampleResult, run_schedule
from pymrst.modules.ad_core import State
from pymrst.modules.ad_core.aquifer import AquiferConnections, AquiferModel
from pymrst.modules.ad_props import init_simple_adi_fluid
from pymrst.params import add_well, make_rock
from pymrst.utils.units import BARSA, CENTI, DARCY, DAY, MILLI, POISE


def build_aquifertest():
    g = compute_geometry(cart_grid([10, 1, 2], [500.0, 100.0, 30.0]))
    g.nodes.coords[:, 2] += 2000.0
    rock = make_rock(g, 100 * MILLI * DARCY, 0.25)
    p_ref = 220 * BARSA
    fluid = init_simple_adi_fluid(
        mu=np.array([0.5, 2.0]) * CENTI * POISE,
        rho=[1000.0, 800.0],
        n=[2, 2],
        c=[4.0e-5 / BARSA, 1.0e-4 / BARSA],
        p_ref=p_ref,
    )
    conn_cells = np.array([g.cells.num - 2, g.cells.num - 1])
    connections = AquiferConnections(
        aquid=np.zeros(conn_cells.size, dtype=int),
        cells=conn_cells,
        j=np.full(conn_cells.size, 2.5e-11),
        c=np.full(conn_cells.size, 1e-5 / BARSA),
        alpha=np.ones(conn_cells.size),
        depth_conn=g.cells.centroids[conn_cells, 2],
        depth_aq=np.array([2015.0, 2015.0]),
        c_aq=np.array([1e-5 / BARSA]),
    )
    aquifer = AquiferModel(connections, np.array([250.0 * BARSA]), np.array([5e7]))
    wells = add_well(None, g, rock, np.array([0]), well_type="bhp",
                     val=150 * BARSA, compi=np.array([0.0, 1.0]), sign=-1,
                     radius=0.1, name="PROD")
    model = GenericBlackOilModel(g, rock, fluid, wells, aquifer=aquifer)
    state = State(
        pressure=np.full(g.cells.num, p_ref),
        sW=np.full(g.cells.num, 0.2),
        qWs=np.zeros(1),
        qOs=np.zeros(1),
        bhp=np.array([150 * BARSA]),
    )
    aquifer.init_state(state)
    return model, state, np.full(6, 20 * DAY)


def run_aquifertest() -> ExampleResult:
    model, state, timesteps = build_aquifertest()
    return run_schedule(
        model, state, timesteps,
        history_keys=("pressure", "sW", "qWs", "qOs", "bhp", "aquifer_p", "aquifer_V"),
    )


if __name__ == "__main__":
    result = run_aquifertest()
    print(f"aquifer final pressure {result.history['aquifer_p'][-1, 0] / BARSA:.2f} bar")
