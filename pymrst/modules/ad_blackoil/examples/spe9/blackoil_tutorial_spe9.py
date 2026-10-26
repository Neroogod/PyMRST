"""MRST ``ad-blackoil/examples/spe9/blackoilTutorialSPE9.m``.

The ninth SPE comparative solution project: a 24x25x15 black-oil
reservoir with one water injector and 25 producers, whose oil-rate targets
drop sharply after 300 days and recover after 360, forcing many producers
onto their bottom-hole-pressure limits. `blackoil_tutorial_spe9` is the
tutorial, section by section, on the AD-OO stack:
- `setupSPE9`, `selectModelFromDeck` and `convertDeckScheduleToMRST` build
  the case; `drsMaxRel`, `dpMaxRel` and `dsMaxAbs` limit the Newton updates;
- `selectLinearSolverAD` picks CPR with a direct pressure solve (27000
  unknowns, sparse AD backend, no AGMG and AMGCL only with `useMEX`);
- the permeability, porosity, initial saturations and dissolved gas, the
  wells, PROD2's controls and limits, and the fluid's relative
  permeabilities, capillary pressures, formation volume factors and
  viscosities are plotted;
- `simulateScheduleAD` runs the schedule with `getPlotAfterStep`'s runtime
  panel;
- the bottom-hole pressures and gas rates of PROD13 and PROD18 are compared
  with ECLIPSE (`compare/SPE9.UNSMRY`), followed by the wells' controls over
  time and the pressure, free gas, dissolved gas and phase distribution
  after the first and last steps.

Without AGMG, and with AMGCL only under MRST's `useMEX` setting (off by
default, and MRST's MEX build fails on Apple M4), the selected solver is CPR
with a direct pressure solve. Its GMRES does not converge within 25
iterations on SPE9, and MRST 2024a gives up in the first report step
("Maximum number of substeps stopped timestep reduction"); the port, whose
CPR reproduces MATLAB's GMRES residuals, does the same. The keyword
`LinearSolver` (a PyMRST addition) runs the schedule with another solver,
e.g. `BackslashSolverAD()`, which is how the rest of the tutorial is
validated (`validate_ad_oo_blackoil_tutorial_spe9.py`).

3-D figures are PyVista windows and 2-D figures matplotlib; `colorbarHist`
draws its histogram colour bar on a matplotlib figure of its own. Without
`plot` nothing is drawn and the run has no after-step function; the
runtime panel it would show does not change the run.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, Sequence

import numpy as np

VIEW = (15, 30)


def blackoil_tutorial_spe9(plot: bool = True, LinearSolver: Any = None) -> SimpleNamespace:
    """MRST `blackoilTutorialSPE9`. Returns the run and the plotted data.

    `LinearSolver` (a PyMRST addition) replaces `selectLinearSolverAD`'s
    choice in the simulation; omitted, the run is MRST's.
    """
    from pymrst.modules.ad_core.ad_oo.simulate_schedule_ad import simulate_schedule_ad
    from pymrst.modules.ad_core.utils.convert_deck_schedule_to_mrst import (
        convert_deck_schedule_to_mrst,
    )
    from pymrst.modules.ad_core.utils.select_linear_solver_ad import select_linear_solver_ad
    from pymrst.modules.ad_core.utils.select_model_from_deck import select_model_from_deck
    from pymrst.modules.test_suite.dataset_functions.spe9 import setup_spe9
    from pymrst.utils.gridtools.grid_logical_indices import grid_logical_indices
    from pymrst.utils.units import BARSA, DARCY, DAY, MILLI, STB, YEAR

    out = SimpleNamespace()
    # Set up model
    G, rock, fluid, deck, state0 = setup_spe9()
    model = select_model_from_deck(G, rock, fluid, deck)
    print(model)
    schedule = convert_deck_schedule_to_mrst(model, deck)
    # Adjust time-step control
    model.drsMaxRel = np.inf
    model.dpMaxRel = .1
    model.dsMaxAbs = .1
    # Configure linear solver
    linsolve = select_linear_solver_ad(model)
    out.linsolve = linsolve

    # Plot the rock permeability
    G, rock = model.G, model.rock
    K = np.asarray(rock.perm)[:, 0]/(MILLI*DARCY)
    if plot:
        _plot_cell_data_with_histogram(G, np.log10(K), K, [.01, 1e4], 51, True,
                                       "SPE9 Horizontal permeability")
    # Plot the grid porosity
    poro = np.asarray(rock.poro).ravel()
    if plot:
        _plot_cell_data_with_histogram(G, poro, poro, [.06, .18], 31, None, "SPE9 Porosity")
    # Plot a single vertical set of cells
    ii, jj = grid_logical_indices(G)[:2]
    out.column = np.flatnonzero((ii == 0) & (jj == 0))
    if plot:
        _plot_column(G, poro, out.column)
    # Plot initial saturation of oil and water
    s = np.asarray(state0["s"])[:, [1, 2, 0]]
    if plot:
        _plot_phase_distribution(G, s, "")
        _plot_rgb_legend()
    # Examine gas dissolved in oil phase
    Rs_sat = model.fluid.rsSat(state0["pressure"])
    out.Rs_frac = np.asarray(state0["rs"])/np.asarray(Rs_sat)
    if plot:
        _plot_3d(G, out.Rs_frac, "Fraction of maximum gas saturation in oil phase - g(p)")
    # Plot the wells
    W = schedule.controls[0].W
    sgn = np.array([w.sign for w in W])
    out.sgn = sgn
    if plot:
        _plot_wells(G, W, sgn)

    # Examine the schedule
    names = [w.name for w in schedule.controls[0].W]
    wno = names.index("PROD2")
    control = np.asarray(schedule.step_control, dtype=np.int64)
    out.P = np.array([float(np.ravel(schedule.controls[c].W[wno].val)[0]) for c in control])
    T = np.cumsum(schedule.step_val)
    if plot:
        _plot_prod2_controls(T/YEAR, -out.P/(STB/DAY))
    # Examine well limits
    out.lims = schedule.controls[0].W[wno].lims
    print(f"Well limits for {schedule.controls[0].W[wno].name}:")
    print(out.lims)

    # Plot relative permeability curves
    f = model.fluid
    sv = np.arange(0, 101)/100
    out.kr = np.column_stack([f.krW(sv), f.krOW(sv), f.krOG(sv), f.krG(sv)])
    # Plot three-phase relative permeability
    x, y = np.meshgrid(sv, sv)
    krO = np.zeros(x.shape)
    for i in range(x.shape[0]):
        xi, yi = x[i, :], y[i, :]
        _, krO[i, :], _ = model.rel_perm_wog(xi, 1 - xi - yi, yi, f)
    krO[x + y > 1] = np.nan
    out.krO = krO
    # Plot capillary pressure curves
    out.pc = np.column_stack([f.pcOG(sv), f.pcOW(1 - sv)])
    # Plot compressibility
    pressure = np.arange(50, 501, 10)*BARSA
    out.pvt = np.column_stack([1/f.bW(pressure), 1/f.bG(pressure), f.pvMultR(pressure),
                               f.muW(pressure), f.muG(pressure)])
    # Plot oil compressibility
    rs = np.arange(0, 321, 25)
    p_g, rs_g = np.meshgrid(pressure, rs)
    rssat = np.zeros(p_g.shape)
    for i in range(p_g.shape[0]):
        rssat[i, :] = f.rsSat(p_g[i, :])
    saturated = rs_g >= rssat
    rs_g = rs_g.astype(float)
    rs_g[saturated] = rssat[saturated]
    out.saturated = saturated
    out.Bo = 1/np.asarray(f.bO(p_g, rs_g, saturated))
    # Plot the viscosity
    out.muO = np.asarray(f.muO(p_g, rs_g, saturated))
    if plot:
        _plot_fluid(sv, out, x, y, pressure, p_g)

    # Simulate the schedule
    model.verbose = False
    fn = None
    if plot:
        from pymrst.modules.ad_core.plotting.get_plot_after_step import get_plot_after_step
        fn = get_plot_after_step(state0, model, schedule, plotWell=False, plotReservoir=False)
    wellsols, states, reports = simulate_schedule_ad(
        state0, model, schedule, LinearSolver=linsolve if LinearSolver is None else LinearSolver,
        afterStepFn=fn)
    out.G, out.model, out.schedule = G, model, schedule
    out.wellsols, out.states, out.reports = wellsols, states, reports

    # Launch interactive plot tool for well curves
    if plot:
        import matplotlib.pyplot as plt
        from pymrst.modules.ad_core.plotting.plot_well_sols import plot_well_sols
        h, _ = plot_well_sols(wellsols, np.cumsum(schedule.step_val))

    # Load comparison data from commercial solver
    from pymrst.modules.ad_blackoil.examples.utils.local_dataset_path import MRST_DATA
    from pymrst.modules.deckformat.resultinput.read_eclipse_summary_un_fmt import (
        read_eclipse_summary_un_fmt,
    )
    compare = MRST_DATA.parents[1]/"modules"/"ad-blackoil"/"examples"/"spe9"/"compare"
    smry, _ = read_eclipse_summary_un_fmt(compare/"SPE9")
    compd = np.arange(smry.data.shape[1])
    out.Tcomp = smry.get(":+:+:+:+", "YEARS", compd).ravel()

    # Set up plotting functions
    if plot and plt.fignum_exists(h.number):
        plt.close(h)
    out.T = np.cumsum(schedule.step_val)/YEAR
    _compare_producers(out, smry, compd, plot)

    # Changing controls
    out.ctrls = np.array([[w.type.lower() == "bhp" for w in ws] for ws in wellsols],
                         dtype=float)
    if plot:
        _plot_controls(out.ctrls, out.T*YEAR/DAY, [w.name for w in wellsols[0]])
        _plot_first_and_last(G, states, state0)
    return out


blackoilTutorialSPE9 = blackoil_tutorial_spe9


def _compare_producers(out: SimpleNamespace, smry: Any, compd: np.ndarray, plot: bool) -> None:
    """The bottom-hole pressures and gas rates of PROD13 and PROD18."""
    from pymrst.modules.ad_core.utils.get_well_output import get_well_output
    from pymrst.utils.units import DAY, FT, PSIA
    names = ["PROD13", "PROD18"]
    column = lambda data: np.asarray(data)[:, 0, 0]
    out.bhp_comp = np.column_stack([smry.get(n, "WBHP", compd).ravel()*PSIA for n in names])
    out.bhp_mrst = np.column_stack([column(get_well_output(out.wellsols, "bhp", n)[0])
                                    for n in names])
    out.gas_comp = np.column_stack([smry.get(n, "WGPR", compd).ravel()*(1000*FT**3/DAY)
                                    for n in names])
    out.gas_mrst = np.abs(np.column_stack([column(get_well_output(out.wellsols, "qGs", n)[0])
                                           for n in names]))
    if not plot:
        return
    import matplotlib.pyplot as plt
    for mrst, comp, ylabel in ((out.bhp_mrst, out.bhp_comp, "Pressure (Pa)"),
                               (out.gas_mrst, out.gas_comp, "Gas rate (m^3/s)")):
        fig = plt.figure()
        for i, name in enumerate(names):
            ax = fig.add_subplot(len(names), 1, i + 1)
            ax.plot(out.T, mrst[:, i], "-b", linewidth=2)
            ax.plot(out.Tcomp, comp[:, i], "ro", linewidth=2)
            ax.set_title(name)
            ax.autoscale(tight=True)
            ax.grid(True)
            ax.set_xlabel("Time (years)")
            ax.set_ylabel(ylabel)
        ax.legend(["MRST", "ECLIPSE"])
    plt.show()


def _plot_controls(ctrls: np.ndarray, t_days: np.ndarray, names: Sequence[str]) -> None:
    """MRST's `pcolor` of each well's control (rate or bhp) per step."""
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    nstep, nw = ctrls.shape
    X = np.tile(np.arange(nw + 1), (nstep, 1)) + .5
    Y = np.tile(t_days[:, None], (1, nw + 1))
    C = np.column_stack([ctrls, ctrls[:, -1]])
    lines = np.array([[0, 0.4470, 0.7410], [0.8500, 0.3250, 0.0980]])
    fig, ax = plt.subplots()
    # MATLAB's `pcolor` colours each cell by its lower-left vertex's value,
    # dropping C's last row and column; `view(90, 90)` puts time across.
    mesh = ax.pcolormesh(Y, X, C[:-1, :-1], cmap=ListedColormap(.55*lines + .45),
                         shading="flat")
    ax.set_yticks([])
    ax.set_xlabel("Time (days)")
    ax.invert_yaxis()
    for k, name in enumerate(names):
        ax.text(Y[0, 1] - 5, (X[0, k] + X[0, k + 1])/2, name, ha="right", va="center",
                fontsize=8)
    bar = fig.colorbar(mesh, ticks=[.25, .75])
    bar.ax.set_yticklabels(["rate", "bhp"])
    plt.show()


def _plot_first_and_last(G: Any, states: Sequence[Any], state0: Any) -> None:
    """Pressure, free gas, dissolved gas and phases after the first and last
    steps, on a shared colour scale."""
    from pymrst.utils.units import BARSA
    p_start, p_end = states[0]["pressure"], states[-1]["pressure"]
    cscale = [np.min(p_end)/BARSA, np.max(p_start)/BARSA]
    _plot_3d(G, p_start/BARSA, "Pressure after first timestep", clim=cscale)
    _plot_3d(G, p_end/BARSA, "Pressure after final timestep", clim=cscale)
    sg0, sg = np.asarray(state0["s"])[:, 2], np.asarray(states[-1]["s"])[:, 2]
    cscale = [0, np.max(sg)]
    _plot_3d(G, sg0, "Free gas after first timestep", clim=cscale)
    _plot_3d(G, sg, "Free gas after final timestep", clim=cscale)
    gasinoil_0 = states[0]["rs"]*np.asarray(states[0]["s"])[:, 1]
    gasinoil = states[-1]["rs"]*np.asarray(states[-1]["s"])[:, 1]
    cscale = [0, np.max(gasinoil_0)]
    _plot_3d(G, gasinoil_0, "Gas in after first timestep", clim=cscale)
    _plot_3d(G, gasinoil, "Gas in oil after final timestep", clim=cscale)
    _plot_phase_distribution(G, np.asarray(states[0]["s"])[:, [1, 2, 0]],
                             "Phase distribution after first timestep")
    _plot_phase_distribution(G, np.asarray(states[-1]["s"])[:, [1, 2, 0]],
                             "Phase distribution after final timstep")


def _plot_fluid(s: np.ndarray, out: SimpleNamespace, x: np.ndarray, y: np.ndarray,
                pressure: np.ndarray, p_g: np.ndarray) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.colors import LightSource
    from pymrst.utils.units import BARSA
    for curves, xlabel, title, legend in (
            (out.kr[:, [0]], "Water saturation", "Water relative permeability curve", None),
            (out.kr[:, [1, 2]], "Oil saturation", "Oil relative permeability curves",
             ["Oil-Water system", "Oil-Gas system"]),
            (out.kr[:, [3]], "Gas saturation", "Gas relative permeability curve", None)):
        plt.figure()
        plt.plot(s, curves, linewidth=2)
        plt.grid(True)
        plt.xlabel(xlabel)
        plt.title(title)
        plt.ylabel("k_r")
        if legend:
            plt.legend(legend, loc="upper left")
    plt.show()
    plt.close("all")
    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")
    shaded = LightSource(azdeg=0, altdeg=90).shade(np.nan_to_num(out.krO), cmap=plt.cm.viridis)
    ax.plot_surface(x, y, out.krO, facecolors=shaded, linewidth=0, antialiased=True)
    ax.set_xlabel("sW")
    ax.set_ylabel("sG")
    ax.set_title("Oil relative permeability")
    ax.view_init(elev=50, azim=150 - 90)
    plt.show()
    plt.close("all")
    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()
    l1, = ax1.plot(s, out.pc[:, 0], linewidth=2, color="C0")
    l2, = ax2.plot(s, out.pc[:, 1], linewidth=2, color="C1")
    ax1.grid(True)
    ax1.legend([l1, l2], ["Oil-Gas capillary pressure", "Oil-Water capillary pressure"],
               loc="lower right")
    ax1.set_xlabel("Oil saturation (Two phase)")
    plt.show()
    plt.close("all")
    pbar = pressure/BARSA
    for column, title, ylabel, xlabel in ((0, "Water formation volume factor", "B_w",
                                           "Pressure [bar]"),
                                          (1, "Gas formation volume factor", "B_g",
                                           "Pressure [bar]"),
                                          (2, "Rock compressibility", "1 + c_r (p - p_s)",
                                           "Pressure [bar]")):
        _line(pbar, out.pvt[:, column], title, ylabel, xlabel)
    _line((p_g/BARSA).T, out.Bo.T, "Oil formation volume factor", "B_o", "Pressure [bar]")
    plt.show()
    plt.close("all")
    _line(pbar, out.pvt[:, 3], "Water viscosity", r"$\mu_w$", "Pressure", ylim=(0, 1.5e-3))
    _line(pbar, out.pvt[:, 4], "Gas viscosity", r"$\mu_g$", "Pressure")
    _line((p_g/BARSA).T, out.muO.T, "Oil viscosity", r"$\mu_o$", "Pressure")
    plt.show()


def _line(x: np.ndarray, y: np.ndarray, title: str, ylabel: str, xlabel: str,
          ylim: Any = None) -> None:
    import matplotlib.pyplot as plt
    plt.figure()
    plt.plot(x, y, linewidth=2)
    plt.grid(True)
    plt.title(title)
    plt.ylabel(ylabel)
    plt.xlabel(xlabel)
    if ylim is not None:
        plt.ylim(ylim)


def _plot_prod2_controls(T_years: np.ndarray, rate_stb: np.ndarray) -> None:
    import matplotlib.pyplot as plt
    plt.figure()
    plt.step(T_years, rate_stb, "o-k", where="post", markersize=6,
             markerfacecolor=[.6, .6, .6])
    plt.xlabel("Time (years)")
    plt.title("Controls for PROD2: oil rate [stb/day]")
    plt.ylim(0, 1600)
    plt.show()


def _plotter(title: str = ""):
    import pyvista as pv
    plotter = pv.Plotter()
    if title:
        plotter.add_title(title)
    return plotter


def _show(plotter: Any) -> None:
    plotter.view_vector(_view_vector(VIEW), viewup=(0, 0, 1))
    plotter.reset_camera()
    plotter.show()


def _view_vector(view: Sequence[float]):
    """MATLAB `view([az, el])` as a camera direction."""
    az, el = np.deg2rad(np.asarray(view, dtype=float))
    return (np.sin(az)*np.cos(el), -np.cos(az)*np.cos(el), np.sin(el))


def _plot_3d(G: Any, data: np.ndarray, title: str, clim: Any = None,
             cells: Any = None) -> None:
    from pymrst.plotting.plot_cell_data import plot_cell_data
    plotter = _plotter(title)
    plot_cell_data(G, data, cells, plotter=plotter, clim=clim, show=False, label=title)
    _show(plotter)


def _plot_cell_data_with_histogram(G: Any, data: np.ndarray, values: np.ndarray,
                                   lim: Sequence[float], nbins: int, logarithmic: Any,
                                   title: str) -> None:
    """`plotCellData` then `colorbarHist(values, lim, 'South', nbins, ...)`."""
    import matplotlib.pyplot as plt
    from pymrst.plotting.colorbar_hist import colorbar_hist
    clim = np.log10(lim) if logarithmic else lim
    _plot_3d(G, data, title, clim=list(clim))
    fig, ax = plt.subplots()
    ax.set_axis_off()
    ax.set_title(title)
    if logarithmic is None:
        colorbar_hist(values, lim, "South", nbins, axes=ax)
    else:
        colorbar_hist(values, lim, "South", nbins, logarithmic, axes=ax)
    plt.show()


def _plot_column(G: Any, poro: np.ndarray, column: np.ndarray) -> None:
    """The porosity with one vertical column of cells outlined in white,
    and the column on its own beside it."""
    import pyvista as pv
    from pymrst.plotting.plot_cell_data import plot_cell_data
    from pymrst.plotting.plot_grid import plot_grid
    plotter = pv.Plotter(shape=(1, 2))
    plotter.subplot(0, 0)
    plot_cell_data(G, poro, plotter=plotter, show=False)
    plot_grid(G, column, face_color="none", edge_color="white", plotter=plotter, show=False)
    plotter.view_vector(_view_vector(VIEW), viewup=(0, 0, 1))
    plotter.subplot(0, 1)
    plot_cell_data(G, poro, column, plotter=plotter, show=False)
    _show(plotter)


def _plot_phase_distribution(G: Any, s: np.ndarray, title: str) -> None:
    """Saturations as colours: oil red, gas green, water blue."""
    _plot_3d(G, np.clip(s, 0, 1), title)


def _plot_rgb_legend() -> None:
    """MATLAB's `patch` triangle with 'FaceColor' 'interp': water blue,
    oil red and gas green at the corners, mixed linearly inside."""
    import matplotlib.pyplot as plt
    corners = np.array([[0, 0], [2, 0], [1, 2*np.sin(np.pi/3)]])
    colours = np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]], dtype=float)
    x, y = np.meshgrid(np.linspace(0, 2, 201), np.linspace(0, corners[2, 1], 175))
    # Barycentric coordinates of each pixel in the triangle.
    T = np.column_stack([corners[0] - corners[2], corners[1] - corners[2]])
    lam = np.linalg.solve(T, np.vstack([x.ravel() - corners[2, 0], y.ravel() - corners[2, 1]]))
    bary = np.vstack([lam, 1 - lam.sum(axis=0)]).T
    image = np.ones((x.size, 4))
    image[:, :3] = np.clip(bary, 0, 1) @ colours
    image[:, 3] = np.all(bary >= -1e-12, axis=1)
    fig, ax = plt.subplots(figsize=(2, 2))
    ax.imshow(image.reshape(*x.shape, 4), origin="lower",
              extent=(0, 2, 0, corners[2, 1]))
    ax.text(0, 0, "W", ha="right")
    ax.text(2, 0, "O", ha="left")
    ax.text(1, corners[2, 1] + .1, "G", ha="center")
    ax.set_axis_off()
    plt.show()


def _plot_wells(G: Any, W: Sequence[Any], sgn: np.ndarray) -> None:
    from pymrst.plotting.plot_grid import plot_grid
    from pymrst.plotting.plot_well import plot_well
    plotter = _plotter()
    plot_grid(G, face_color="none", plotter=plotter, show=False)
    plot_well(G, [w for w, s in zip(W, sgn) if s > 0], plotter, fontsize=0, color="b")
    plot_well(G, [w for w, s in zip(W, sgn) if s < 0], plotter, fontsize=0, color="r")
    _show(plotter)


__all__ = ["blackoilTutorialSPE9", "blackoil_tutorial_spe9"]


if __name__ == "__main__":
    from pymrst.params.gravity.gravity import isolated_gravity

    def test_the_tutorial_picks_cpr_with_a_direct_pressure_solve():
        """27000 unknowns on the sparse backend; AMGCL needs `useMEX`."""
        from pymrst.modules.ad_core.solvers.concrete_solvers import BackslashSolverAD
        from pymrst.modules.ad_core.solvers.cpr_solver_ad import CPRSolverAD
        from pymrst.modules.ad_core.utils.select_linear_solver_ad import (
            select_linear_solver_ad,
        )
        from pymrst.modules.ad_core.utils.select_model_from_deck import select_model_from_deck
        from pymrst.modules.test_suite.dataset_functions.spe9 import setup_spe9
        with isolated_gravity():
            G, rock, fluid, deck, _ = setup_spe9()
            model = select_model_from_deck(G, rock, fluid, deck)
        solver = select_linear_solver_ad(model)
        assert type(solver) is CPRSolverAD
        assert type(solver.ellipticSolver) is BackslashSolverAD
        print("ok test_the_tutorial_picks_cpr_with_a_direct_pressure_solve")

    def test_the_view_is_matlabs():
        """`view(0, 90)` looks straight down; `view(0, 0)` along +y."""
        np.testing.assert_allclose(_view_vector((0, 90)), (0, 0, 1), atol=1e-15)
        np.testing.assert_allclose(_view_vector((0, 0)), (0, -1, 0), atol=1e-15)
        print("ok test_the_view_is_matlabs")

    test_the_tutorial_picks_cpr_with_a_direct_pressure_solve()
    test_the_view_is_matlabs()
    print("\nAll tests passed!")
