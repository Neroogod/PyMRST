"""Port of MRST ``immiscibleTimeIntegrationExample.m``.

PyMRST currently implements the fully implicit path. The explicit/AIM variants
in MRST rely on time-discretization backends that are not part of PyMRST yet,
so this example exposes the same 1D heterogeneous waterflood and a simple CFL
diagnostic for the implicit solution.
"""

from __future__ import annotations

import numpy as np

from pymrst.gridprocessing import cart_grid, compute_geometry
from pymrst.modules.ad_blackoil.models import GenericBlackOilModel
from pymrst.modules.ad_blackoil.examples.utils.schedule import ExampleResult, run_schedule
from pymrst.modules.ad_core import State, rampup_timesteps
from pymrst.modules.ad_props import init_simple_adi_fluid
from pymrst.params import fluxside, make_rock, pore_volume, pside
from pymrst.utils.units import BARSA, DARCY, DAY


def build_immiscible_time_integration_example(ncells: int = 75):
    length = 1000.0
    g = compute_geometry(cart_grid([ncells, 1, 1], [length, 1.0, 1.0]))
    x = g.cells.centroids[:, 0] / length
    poro = np.full(g.cells.num, 0.5)
    for lo, hi, val in [(0.20, 0.30, 0.05), (0.45, 0.55, 0.1), (0.70, 0.80, 0.05)]:
        poro[(x > lo) & (x < hi)] = val
    rock = make_rock(g, 1.0 * DARCY, poro[:, None])
    fluid = init_simple_adi_fluid(mu=[1e-3, 1e-3], rho=[1000.0, 800.0], n=[1, 1])
    model = GenericBlackOilModel(g, rock, fluid, [], tol_cnv=1e-8, tol_mb=1e-12)
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
    timesteps = rampup_timesteps(total_time, total_time / 50.0, 8)
    return model, state, timesteps, bc, poro


def estimate_saturation_cfl(result: ExampleResult, timesteps) -> np.ndarray:
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
    return result, estimate_saturation_cfl(result, timesteps)


if __name__ == "__main__":
    result, cfl = run_immiscible_time_integration_example()
    print(f"implicit run: {len(result.states)} states, max dS/day {cfl.max():.3e}")
