"""MRST ``ad-core/examples/wenoExampleAD.m``.

A quarter five-spot on 50x50 cells of a 100 m x 100 m square (0.1 darcy, 30 %
porosity, two phases with linear relative permeabilities, equal viscosities
and densities): water injected at twice the pore volume a year in the first
cell, production at 50 bar from the last, 300 equal steps over one year. It is
simulated four ways: fully implicit with single-point upwind (SPU) and with
WENO (`setWENODiscretization`), and adaptive implicit (`setTimeDiscretization`
'AIM', initial step 0.02 days) with each. WENO keeps the front sharper; the
saturation CFL of the first SPU state shows where an explicit scheme would be
stable.

With `plot`, MRST's figures: the CFL number with streamlines from `pollock`
and the cells above CFL 1 outlined, the four saturation fields step by step,
the implicit and adaptive implicit fronts at step 80 for each upwinding, and
the water rate of the producer.

`weno_example_ad_ad` (in `weno_example_ad_ad.py`) is PyMRST's earlier version
on the legacy models, kept for its callers.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import numpy as np

from pymrst.gridprocessing import cart_grid, compute_geometry
from pymrst.modules.ad_blackoil.models.generic_black_oil_model import GenericBlackOilModel
from pymrst.modules.ad_core.ad_oo.simulate_schedule_ad import simulate_schedule_ad
from pymrst.modules.ad_core.utils.estimate_saturation_cfl import estimate_saturation_cfl
from pymrst.modules.ad_core.utils.set_time_discretization import set_time_discretization
from pymrst.modules.ad_core.utils.set_weno_discretization import set_weno_discretization
from pymrst.modules.ad_core.utils.simple_schedule import simple_schedule
from pymrst.modules.ad_props import init_simple_adi_fluid
from pymrst.params.rock import make_rock
from pymrst.params.wells_and_bc import add_well
from pymrst.solvers.init_res_sol import init_res_sol
from pymrst.utils.units import BARSA, CENTI, DARCY, DAY, POISE, YEAR

NX, NY = 50, 50
PDIMS = [100.0, 100.0]
NSTEPS = 300
CFL_FLOOR = 0.01
FRONT_STEP = 80


def weno_example_ad(plot: bool = False) -> SimpleNamespace:
    """MRST `wenoExampleAD`: the four runs, their schedule and the CFL field."""
    G = compute_geometry(cart_grid([NX, NY], PDIMS))
    rock = make_rock(G, 0.1*DARCY, 0.3)
    fluid = init_simple_adi_fluid(n=[1, 1], mu=[1*CENTI*POISE, 1*CENTI*POISE],
                                  rho=[100, 100], phases="wo")
    model = GenericBlackOilModel(G, rock, fluid, water=True, oil=True, gas=False)
    time = 1*YEAR
    irate = 2*np.sum(model.operators.pv)/time
    W = add_well(None, G, rock, np.array([0]), compi=np.array([1.0, 0.0]), val=irate,
                 well_type="rate")
    W = add_well(W, G, rock, np.array([G.cells.num - 1]), compi=np.array([1.0, 0.0]),
                 val=50*BARSA, well_type="bhp")
    dt = np.full(NSTEPS, time/NSTEPS)
    schedule = simple_schedule(dt, W=W)
    state0 = init_res_sol(G, 100*BARSA, np.array([0.0, 1.0]))
    ws, states, report = simulate_schedule_ad(state0, model, schedule)
    forces = model.get_driving_forces(schedule.controls[0])
    cfl = estimate_saturation_cfl(model, states[0], np.max(schedule.step_val), forces=forces)
    model_weno = set_weno_discretization(model)
    ws_weno, states_weno, report_weno = simulate_schedule_ad(state0, model_weno, schedule)
    model_e = set_time_discretization(model, "AIM", initialstep=0.02*DAY, verbose=True)
    ws_e, states_e, report_e = simulate_schedule_ad(state0, model_e, schedule)
    model_weno_expl = set_time_discretization(model_weno, "AIM", initialstep=0.02*DAY,
                                              verbose=True)
    ws_weno_ex, states_weno_ex, report_weno_ex = simulate_schedule_ad(
        state0, model_weno_expl, schedule)
    out = SimpleNamespace(G=G, W=W, schedule=schedule, cfl=cfl,
                          ws=ws, states=states, report=report,
                          ws_weno=ws_weno, states_weno=states_weno, report_weno=report_weno,
                          ws_e=ws_e, states_e=states_e, report_e=report_e,
                          ws_weno_ex=ws_weno_ex, states_weno_ex=states_weno_ex,
                          report_weno_ex=report_weno_ex)
    if plot:
        _plot(out)
    return out


def _cart(G: Any, v: np.ndarray) -> np.ndarray:
    return np.reshape(np.asarray(v, dtype=float), G.cartDims, order="F").T


def _plot(out: SimpleNamespace) -> None:
    import matplotlib.pyplot as plt

    from pymrst.modules.ad_core.plotting.plot_well_sols import plot_well_sols
    from pymrst.modules.streamlines.pollock import pollock
    G = out.G
    extent = (0, PDIMS[0], 0, PDIMS[1])
    fig, ax = plt.subplots()
    im = ax.imshow(_cart(G, np.log10(out.cfl + CFL_FLOOR)), origin="lower", extent=extent,
                   cmap="jet")
    ticks = np.array([CFL_FLOOR, 0.1, 1, 10, 30])
    cb = fig.colorbar(im, ticks=np.log10(ticks))
    cb.set_ticklabels([f"{t:g}" for t in ticks])
    flux = np.asarray(out.states[0]["flux"])
    seed = np.arange(NX, NX*NY + 1, NX - 1) - 1
    for reverse in (False, True):
        for line in pollock(G, flux, seed, substeps=1, reverse=reverse):
            ax.plot(line.coords[:, 0], line.coords[:, 1], "k", linewidth=0.5)
    ax.contour(_cart(G, out.cfl > 1).astype(float), levels=[0.5], colors="w",
               extent=extent)
    ax.set_aspect("equal")
    ax.set_axis_off()
    runs = (("FIM SPU", out.states), ("FIM WENO", out.states_weno),
            ("AIM SPU", out.states_e), ("AIM WENO", out.states_weno_ex))
    fig, axes = plt.subplots(2, 2)
    for i in range(len(out.states)):
        for a, (title, states) in zip(axes.ravel(), runs):
            a.clear()
            a.imshow(_cart(G, np.asarray(states[i]["s"])[:, 0]), origin="lower",
                     extent=extent)
            a.set_title(title)
            a.set_aspect("equal")
        plt.pause(0.001)
    cval = np.concatenate([0.5*(np.linspace(0, 1, 21)[:-1] + np.linspace(0, 1, 21)[1:]), [1]])
    for impl, expl in ((out.states, out.states_e), (out.states_weno, out.states_weno_ex)):
        fig, a = plt.subplots()
        a.contourf(_cart(G, np.asarray(impl[FRONT_STEP - 1]["s"])[:, 0]), cval,
                   extent=extent, cmap="winter_r")
        a.contour(_cart(G, np.asarray(expl[FRONT_STEP - 1]["s"])[:, 0]), cval[1:-1],
                  colors="k", extent=extent)
        for w in out.W:
            c = G.cells.centroids[np.atleast_1d(w.cells)[0]]
            a.plot(c[0], c[1], "o", markersize=8, markerfacecolor="r", markeredgecolor="k")
        a.set_aspect("equal")
        a.set_axis_off()
    plot_well_sols([out.ws, out.ws_e, out.ws_weno, out.ws_weno_ex],
                   np.cumsum(out.schedule.step_val),
                   datasetnames=["FIM-SPU", "AIM-SPU", "FIM-WENO", "AIM-WENO"],
                   field="qWs", SelectedWells=[2])


wenoExampleAD = weno_example_ad

__all__ = ["weno_example_ad", "wenoExampleAD"]


if __name__ == "__main__":
    import contextlib
    import io

    def test_four_runs_and_weno_sharpens_the_front():
        with contextlib.redirect_stdout(io.StringIO()):
            out = weno_example_ad()
        for name in ("states", "states_weno", "states_e", "states_weno_ex"):
            assert len(getattr(out, name)) == NSTEPS, name
        neighbors = out.G.faces.neighbors[np.all(out.G.faces.neighbors >= 0, axis=1)]
        jumps = [np.abs(np.diff(np.asarray(s[FRONT_STEP - 1]["s"])[neighbors, 0], axis=1)).max()
                 for s in (out.states, out.states_weno)]
        assert jumps[1] > 1.5*jumps[0], jumps
        assert out.cfl.shape == (out.G.cells.num,) and out.cfl.max() > 1
        print("ok test_four_runs_and_weno_sharpens_the_front")

    test_four_runs_and_weno_sharpens_the_front()
    print("\nAll tests passed!")
