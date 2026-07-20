"""Reduced port of MRST ``multisegmentWellExample.m``."""

from __future__ import annotations

import numpy as np

from pymrst.gridprocessing import cart_grid, compute_geometry
from pymrst.modules.ad_blackoil.models import GenericBlackOilModel
from pymrst.modules.ad_blackoil.examples.utils.schedule import ExampleResult, run_schedule
from pymrst.modules.ad_core import State
from pymrst.modules.ad_props import init_simple_adi_fluid
from pymrst.params import add_well, make_rock
from pymrst.params.wells_and_bc import MultisegmentWell, WellSegment
from pymrst.utils.units import BARSA, CENTI, DARCY, DAY, MILLI, POISE


def build_multisegment_well_example(rate: float = 1e-6):
    g = compute_geometry(cart_grid([3, 1, 1], [300.0, 50.0, 20.0]))
    rock = make_rock(g, 100 * MILLI * DARCY, 0.25)
    fluid = init_simple_adi_fluid(
        mu=np.array([1.0, 5.0]) * CENTI * POISE,
        rho=[1000.0, 800.0],
        n=[2, 2],
        c=[1e-6 / BARSA, 1e-4 / BARSA],
        p_ref=200 * BARSA,
    )
    wells = add_well(None, g, rock, np.array([0, 1]), well_type="rate",
                     val=rate, compi=np.array([1.0, 0.0]), sign=1,
                     radius=0.1, name="MSI")
    wells[0].multisegment = MultisegmentWell([
        WellSegment(0, -1, depth=1000.0, diameter=0.2, volume=10.0),
        WellSegment(1, 0, length=10.0, depth=1000.0, diameter=0.2,
                    roughness=1e-5, volume=10.0),
        WellSegment(2, 1, length=10.0, depth=1000.0, diameter=0.2,
                    roughness=1e-5, volume=10.0),
    ], np.array([1, 2]))
    wells = add_well(wells, g, rock, np.array([2]), well_type="bhp",
                     val=195 * BARSA, compi=np.array([0.0, 1.0]),
                     sign=-1, radius=0.1, name="PROD")
    model = GenericBlackOilModel(g, rock, fluid, wells, tol_well=1e-2)
    state = State(
        pressure=np.full(g.cells.num, 200 * BARSA),
        sW=np.full(g.cells.num, 0.2),
        qWs=np.array([rate, 0.0]),
        qOs=np.array([0.0, 0.0]),
        bhp=np.array([200 * BARSA, 195 * BARSA]),
    )
    timesteps = np.array([1.0 * DAY])
    return model, state, timesteps


def run_multisegment_well_example() -> ExampleResult:
    model, state, timesteps = build_multisegment_well_example()
    return run_schedule(
        model, state, timesteps,
        history_keys=("pressure", "sW", "qWs", "qOs", "bhp", "ms_pN", "ms_rW", "ms_rO", "ms_vmix"),
    )


if __name__ == "__main__":
    result = run_multisegment_well_example()
    print(f"MS well pN: {result.history['ms_pN'][-1] / BARSA}")
