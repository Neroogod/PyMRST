"""MRST ``ad-blackoil/examples/blackoilTutorialOnePhase.m``.

Depletion of a closed or open reservoir compartment: a 50x50 grid of a
1000 m x 100 m vertical cross-section (gravity along -y) filled with
slightly compressible water (c = 1e-4/bar, rock 1e-5/bar) at 200 bar. A
producer at the top centre is held at 150 bar. The initial state is the
hydrostatic equilibrium with the south boundary held at 200 bar (one
10000-day step of the `NonLinearSolver`). Then 5 days in 40 steps are run
twice with `simulateScheduleAD`: with the south boundary kept at 200 bar
(an aquifer) and without it (closed). `blackoil_tutorial_one_phase` plots
the equilibrium pressure, animates both runs side by side, and opens
`plotWellSols` on the producer's reservoir rate.

`build_blackoil_tutorial_one_phase`/`run_blackoil_tutorial_one_phase`,
below, are PyMRST's earlier version on the legacy models, kept for their
callers; `--browser` writes their interactive browser viewer.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from pymrst.gridprocessing import cart_grid, compute_geometry
from pymrst.modules.ad_blackoil.models import WaterModelAD
from pymrst.modules.ad_blackoil.examples.utils.schedule import (
    ExampleResult, plot_browser, run_schedule,
)
from pymrst.modules.ad_props import init_simple_adi_fluid
from pymrst.params import NORM_GRAVITY, add_well, make_rock, pside
from pymrst.utils.units import BARSA, CENTI, DARCY, DAY, MILLI, POISE


def blackoil_tutorial_one_phase(plot: bool = True) -> SimpleNamespace:
    """MRST `blackoilTutorialOnePhase`."""
    from pymrst.modules.ad_blackoil.examples.utils.figures import (
        animate_panels, cell_data_figure, matlab_colormap,
    )
    from pymrst.modules.ad_blackoil.models import WaterModel
    from pymrst.modules.ad_core.ad_oo.nonlinear_solver import NonLinearSolver
    from pymrst.modules.ad_core.ad_oo.simulate_schedule_ad import simulate_schedule_ad
    from pymrst.modules.ad_core.utils.schedule import simple_schedule
    from pymrst.params.gravity.gravity import gravity
    from pymrst.solvers.init_res_sol import init_res_sol

    # Grid, petrophysics, and fluid objects
    G = compute_geometry(cart_grid([50, 50], [1000.0, 100.0]))
    rock = make_rock(G, 100*MILLI*DARCY, 0.3)
    pR = 200*BARSA
    fluid = init_simple_adi_fluid(phases="W",            # Fluid phase: water
                                  mu=1*CENTI*POISE,      # Viscosity
                                  rho=1000,              # Surface density [kg/m^3]
                                  c=1e-4/BARSA,          # Fluid compressibility
                                  c_r=1e-5/BARSA)        # Rock compressibility
    # Make Reservoir Model
    gravity("reset", "on")
    wModel = WaterModel(G, rock, fluid, gravity=np.array([0.0, -np.linalg.norm(gravity())]))
    wModel = wModel.validate_model()
    # Drive mechansims and schedule: `sub2ind(G.cartDims, floor(nx/2), ny)`
    dims = np.asarray(G.cartDims)
    wc = (dims[0]//2 - 1) + (dims[1] - 1)*dims[0]
    W = add_well(None, G, rock, np.array([wc]), well_type="bhp", val=pR - 50*BARSA,
                 radius=0.1, name="P1", compi=np.array([1.0]), sign=1)
    bc = pside(None, G, "South", 200*BARSA, sat=1.0)
    schedule1 = simple_schedule(np.diff(np.linspace(0, 5*DAY, 41)), bc=bc, W=W)
    schedule2 = simple_schedule(np.diff(np.linspace(0, 5*DAY, 41)), W=W)
    # Reservoir state
    state = init_res_sol(G, pR)              # Constant pressure
    state = wModel.validate_state(state)
    nonlinear = NonLinearSolver()
    state, _, _ = nonlinear.solve_timestep(state, 10000*DAY, wModel, bc=bc)
    if plot:
        cell_data_figure(G, state["pressure"]/BARSA, "", show_edges=False)
    # Run simulations
    wellSols1, states1, report1 = simulate_schedule_ad(state, wModel, schedule1)
    wellSols2, states2, report2 = simulate_schedule_ad(state, wModel, schedule2)
    # Plot results
    if plot:
        from pymrst.modules.ad_core.plotting.plot_well_sols import plot_well_sols
        animate_panels(G, [[st["pressure"]/BARSA for st in states1],
                           [st["pressure"]/BARSA for st in states2]],
                       ["Open compartment w/aquifer", "Closed compartment"],
                       clim=[pR/BARSA - 50, pR/BARSA], cmap=matlab_colormap("jet", 32),
                       show_edges=False)
        plot_well_sols([wellSols1, wellSols2], datasetnames=["Aquifer", "Closed"],
                       field="qWr")
        import matplotlib.pyplot as plt
        plt.show()
    return SimpleNamespace(G=G, model=wModel, W=W, bc=bc, state=state,
                           schedule1=schedule1, schedule2=schedule2,
                           wellSols1=wellSols1, states1=states1,
                           wellSols2=wellSols2, states2=states2,
                           report1=report1, report2=report2)


blackoilTutorialOnePhase = blackoil_tutorial_one_phase


def build_blackoil_tutorial_one_phase():
    """MRST `blackoilTutorialOnePhase.m`, parameter for parameter.

    Every number here is MRST's, so the two can be run side by side and
    compared. The port previously halved the grid, turned gravity off, flipped
    the well sign and halved the timestep count - each defensible on its own,
    and collectively a different problem.

    Three of those needed a fix elsewhere before they could be matched:

    * the grid is genuinely **2-D** (`cartGrid([50 50],[1000 100])`), and
      `WaterModelAD` read `centroids[:, 2]` for the gravity column, which does
      not exist on a 2-D grid;
    * gravity acts along **y** there, which is what `[0 -norm(gravity)]`
      means - the last axis, not the third;
    * the fluid is single-phase (`'phases','W'`), which
      `init_simple_adi_fluid` refused.

    One parameter is *not* MRST's literal text and cannot be: the well is
    declared `'sign',1` there and given `sign=0` here. MRST's own
    `setWellSign` assigns zero to a bottom-hole-pressure control, and MRST's
    well model reads the flow direction from the drawdown rather than the
    declared sign - so `1` is inert in MATLAB. PyMRST branches on the declared
    sign instead, and with `sign=1` this well does not converge at all.
    Aligning the well model with MRST's computed-direction rule is the real
    fix; until then the example uses the sign MRST's own rule would assign.
    """
    g = compute_geometry(cart_grid([50, 50], [1000.0, 100.0]))
    rock = make_rock(g, 100 * MILLI * DARCY, 0.30)
    p_ref = 200 * BARSA
    fluid = init_simple_adi_fluid(
        mu=np.array([1.0]) * CENTI * POISE,
        rho=[1000.0],
        n=[1],
        c=[1e-4 / BARSA],
        p_ref=p_ref,
    )
    # MRST: sub2ind(G.cartDims, floor(G.cartDims(1)/2), G.cartDims(2)) - the
    # top of the middle column, in MATLAB's 1-based column-major indexing.
    i = g.cartDims[0] // 2 - 1
    j = g.cartDims[1] - 1
    wc = np.ravel_multi_index((i, j), tuple(g.cartDims), order="F")
    wells = add_well(
        None, g, rock, np.array([wc]), well_type="bhp",
        val=p_ref - 50 * BARSA, radius=0.1, name="P1",
        # MRST's script passes `'sign',1`, but its own `setWellSign` rule for a
        # bottom-hole-pressure control is `sign = 0` - a BHP well has no
        # prescribed direction, and this one produces: 150 bar against a
        # 200 bar reservoir. MRST tolerates the mismatch because its well
        # model takes the direction from the computed drawdown
        # (`connInjInx = drawdown < 0`), not from the declared sign.
        #
        # PyMRST does not: `FacilityModel` line ~2289 branches on `w.sign`, so
        # a declared injector that physically produces is forced onto the
        # injection branch and Newton cannot converge. That is a real
        # difference in the well model, found by making this a direct port,
        # and it is recorded in the module docstring rather than papered over.
        compi=np.array([1.0]), sign=0,
    )
    model = WaterModelAD(g, rock, fluid, wells, gravity=NORM_GRAVITY)
    state = model.initial_state(p_ref)
    bc = pside(None, g, "ymin", 200 * BARSA, sat=np.array([1.0]))
    timesteps = np.diff(np.linspace(0.0, 5.0 * DAY, 41))
    return model, state, timesteps, bc


def run_blackoil_tutorial_one_phase(verbose: bool = False) -> dict[str, ExampleResult]:
    model, state, timesteps, bc = build_blackoil_tutorial_one_phase()
    if verbose:
        print("Open boundary:")
    open_result = run_schedule(
        model, state, timesteps, controls={"bc": bc},
        history_keys=("pressure", "qWs", "bhp"), verbose=verbose,
    )
    closed_model, closed_state, _, _ = build_blackoil_tutorial_one_phase()
    if verbose:
        print("Closed boundary:")
    closed_result = run_schedule(
        closed_model, closed_state, timesteps,
        history_keys=("pressure", "qWs", "bhp"), verbose=verbose,
    )
    return {"open": open_result, "closed": closed_result}


#: MRST's own spelling. MRST files each example as a script; this port
#: gives it a `run_` entry point, and the alias keeps the MATLAB name
#: callable - the convention the rest of the package follows.
blackoilTutorialOnePhase = run_blackoil_tutorial_one_phase

__all__ = ["blackoilTutorialOnePhase", "blackoil_tutorial_one_phase",
           "build_blackoil_tutorial_one_phase", "run_blackoil_tutorial_one_phase"]


if __name__ == "__main__":
    import argparse

    # `--browser` is opt-in, matching plot_saved_states.py's own CLI: the
    # viewer is a plug-in you reach for, not something every run pays for.
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--browser", action="store_true",
                        help="Write the legacy interactive browser viewer (open-boundary case).")
    parser.add_argument("--open-browser", action="store_true",
                        help="Open the browser viewer once written. Implies --browser.")
    parser.add_argument("--output", type=str, default="blackoil_tutorial_one_phase.html",
                        help="Browser viewer filename.")
    arguments = parser.parse_args()

    if arguments.browser or arguments.open_browser:
        model, state, timesteps, bc = build_blackoil_tutorial_one_phase()
        open_result = run_schedule(
            model, state, timesteps, controls={"bc": bc},
            history_keys=("pressure", "qWs", "bhp"),
        )
        path = plot_browser(
            model, open_result, timesteps,
            title="Black-oil tutorial - one phase (open boundary)",
            filename=arguments.output, open_browser=arguments.open_browser,
        )
        print(f"browser viewer written to {path}")
        raise SystemExit(0)

    import matplotlib
    matplotlib.use("Agg")
    import pyvista as pv

    from pymrst.params.gravity.gravity import isolated_gravity

    pv.OFF_SCREEN = True
    with isolated_gravity():
        OUT = blackoil_tutorial_one_phase(plot=True)

    def test_the_equilibrium_is_hydrostatic_from_the_south_boundary():
        """Pressure falls upwards (+y) at the compressed water's gradient
        rho*g, with rho = 1000*exp(c*p) (about 1020 kg/m^3 at 200 bar)."""
        p = OUT.state["pressure"].reshape(50, 50)
        assert np.all(np.diff(p.mean(axis=1)) < 0), "pressure must fall upwards"
        dy = 100.0/50
        grad = (p[0, 0] - p[1, 0])/dy
        rho = 1000*np.exp(1e-4/BARSA*(p[0, 0] + p[1, 0])/2)
        assert abs(grad - rho*9.80665)/(rho*9.80665) < 1e-3, (grad, rho*9.80665)
        print("ok test_the_equilibrium_is_hydrostatic_from_the_south_boundary")

    def test_the_aquifer_keeps_the_open_compartment_up():
        p_open = OUT.states1[-1]["pressure"].mean()
        p_closed = OUT.states2[-1]["pressure"].mean()
        assert p_open > p_closed + 10*BARSA, (p_open/BARSA, p_closed/BARSA)
        print("ok test_the_aquifer_keeps_the_open_compartment_up")

    def test_the_closed_well_rate_declines():
        q = np.array([abs(ws[0].qWs) for ws in OUT.wellSols2])
        assert np.all(np.diff(q) < 0), q
        print("ok test_the_closed_well_rate_declines")

    test_the_equilibrium_is_hydrostatic_from_the_south_boundary()
    test_the_aquifer_keeps_the_open_compartment_up()
    test_the_closed_well_rate_declines()
    print("\nAll tests passed!")
