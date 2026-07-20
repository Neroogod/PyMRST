"""Port of MRST ``blackoilTutorialOnePhase.m``."""

from __future__ import annotations

import numpy as np

from pymrst.gridprocessing import cart_grid, compute_geometry
from pymrst.modules.ad_blackoil.models import WaterModel
from pymrst.modules.ad_blackoil.examples.utils.schedule import ExampleResult, run_schedule
from pymrst.modules.ad_props import init_simple_adi_fluid
from pymrst.params import add_well, make_rock, pside
from pymrst.utils.units import BARSA, CENTI, DARCY, DAY, MILLI, POISE


def build_blackoil_tutorial_one_phase():
    g = compute_geometry(cart_grid([25, 25, 1], [1000.0, 100.0, 10.0]))
    rock = make_rock(g, 100 * MILLI * DARCY, 0.30)
    p_ref = 200 * BARSA
    fluid = init_simple_adi_fluid(
        mu=np.array([1.0, 1.0]) * CENTI * POISE,
        rho=[1000.0, 800.0],
        n=[1, 1],
        c=[1e-4 / BARSA, 0.0],
        p_ref=p_ref,
    )
    i = g.cartDims[0] // 2
    j = g.cartDims[1] - 1
    wc = np.ravel_multi_index((i, j, 0), g.cartDims, order="F")
    wells = add_well(
        None, g, rock, np.array([wc]), well_type="bhp",
        val=p_ref - 50 * BARSA, radius=0.1, name="P1",
        compi=np.array([1.0]), sign=-1,
    )
    model = WaterModel(g, rock, fluid, wells, gravity=0.0)
    state = model.initial_state(p_ref)
    bc = pside(None, g, "ymin", p_ref, sat=np.array([1.0]))
    timesteps = np.diff(np.linspace(0.0, 5.0 * DAY, 21))
    return model, state, timesteps, bc


def run_blackoil_tutorial_one_phase() -> dict[str, ExampleResult]:
    model, state, timesteps, bc = build_blackoil_tutorial_one_phase()
    open_result = run_schedule(
        model, state, timesteps, controls={"bc": bc},
        history_keys=("pressure", "qWs", "bhp"),
    )
    closed_model, closed_state, _, _ = build_blackoil_tutorial_one_phase()
    closed_result = run_schedule(
        closed_model, closed_state, timesteps,
        history_keys=("pressure", "qWs", "bhp"),
    )
    return {"open": open_result, "closed": closed_result}


if __name__ == "__main__":
    out = run_blackoil_tutorial_one_phase()
    for name, result in out.items():
        p = result.history["pressure"][-1] / BARSA
        print(f"{name}: final pressure range {p.min():.2f}-{p.max():.2f} bar")
