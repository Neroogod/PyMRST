"""MRST ``ad-blackoil/examples/egg/eggExample.m``.

The two-phase oil-water Egg model (realization 0 by default): `setupEGG`
reads the deck, `initEclipseProblemAD` (with the grid from `setupEGG` and no
timestep strategy) builds the model, schedule, equilibrated initial state
and nonlinear solver, and `simulateScheduleAD` runs the ten-year schedule.
`egg_example` plots the injectors' bottom-hole pressures, the producers' oil
and water rates and water cut, animates the pressure and water saturation
with the wells, and opens `plotWellSols` and `plotToolbar`. Without `plot`
nothing is drawn.

`run_egg_example`, below, is PyMRST's earlier version on the legacy models,
kept for its callers. Its notes follow.

MRST's script, in full::

    realization = 0;
    [G, rock, fluid, deck] = setupEGG('realization', realization);
    [state, model, schedule, nonlinear] = ...
        initEclipseProblemAD(deck, 'G', G, 'TimestepStrategy', 'none');
    [wellSols, states, report] = ...
        simulateScheduleAD(state, model, schedule, 'NonLinearSolver', nonlinear);

    T    = convertTo(cumsum(schedule.step.val), year);
    W    = schedule.control(1).W;
    inj  = find(vertcat(W.sign) > 0);
    prod = find(vertcat(W.sign) < 0);

    bhp  = getWellOutput(wellSols, 'bhp',  inj);    % injector BHP
    orat = getWellOutput(wellSols, 'qOs', prod);    % producer oil rate
    wrat = getWellOutput(wellSols, 'qWs', prod);    % producer water rate
                                                    % water cut = |w|/|w+o|

That is what this runs. The previous version was not a port of it: it read the
deck with a bespoke Egg-only parser, built the grid from `cart_grid` with the
cell sizes written in by hand rather than from the deck, and initialised
pressure from a hydrostatic formula of its own instead of from `EQUIL`. Each
of those produced a plausible Egg-*like* case; none of them was the deck's.

Making it MRST's required the deck reader to grow the three sections it had
listed as skipped - GRID, SOLUTION and SCHEDULE - so `read_eclipse_deck` now
returns the whole deck, `setup_egg` is a transcription of `setupEGG.m`, and
this is a transcription of `eggExample.m`.

**One difference from MRST is structural and cannot be hidden.**
`simulateScheduleAD` returns `wellSols` as its first output; PyMRST's schedule
runner returns `(states, report)` and keeps the well unknowns on the *state*
as `bhp`, `qWs`, `qOs` - that is where `init_state_ad` seeds their
derivatives. The `wellSol` rows a state carries are the ones
`init_well_sol_ad` seeded and are not updated by the solve. So `well_sols`
below rebuilds MRST's structure from the solved state - through
`ad_core.utils.well_sol_from_state`, which `get_plot_after_step` shares - and
`get_well_output` then reads it exactly as the MATLAB script does. Reading the
stale `wellSol` rows instead gives every producer a rate of zero for the whole
run.

The case is 18553 cells, 12 wells and 123 report steps over 3600 days. A
smaller `nsteps` runs the leading part of the same schedule; it does not
change any parameter.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any, Dict, List, Optional

import numpy as np

from pymrst.modules.ad_core.initialization.init_eclipse_problem_ad import (
    init_eclipse_problem_ad_ad,
)
from pymrst.modules.ad_core.utils.get_well_output import get_well_output_ad
from pymrst.modules.ad_core.utils.schedule import Schedule
from pymrst.modules.ad_core.utils.well_sol_from_state import (
    well_sol_from_state,
)
from pymrst.modules.test_suite.dataset_functions.egg import setup_egg
from pymrst.utils.units import BARSA, DAY, YEAR



def egg_example(plot: bool = True, realization: int = 0) -> SimpleNamespace:
    """MRST `eggExample`."""
    from pymrst.modules.ad_core.ad_oo.simulate_schedule_ad import simulate_schedule_ad
    from pymrst.modules.ad_core.initialization.init_eclipse_problem_ad import (
        init_eclipse_problem_ad,
    )
    from pymrst.modules.ad_core.utils.get_well_output import get_well_output

    G, rock, fluid, deck, _ = setup_egg(realization)
    state, model, schedule, nonlinear = init_eclipse_problem_ad(deck, G=G,
                                                                TimestepStrategy="none")
    # Run simulation
    wellSols, states, report = simulate_schedule_ad(state, model, schedule,
                                                    NonLinearSolver=nonlinear)
    # Plot the injector BHP and the well oil and water rates
    T = np.cumsum(schedule.step_val)/YEAR
    W = schedule.controls[0].W
    inj = np.flatnonzero([w.sign > 0 for w in W])
    prod = np.flatnonzero([w.sign < 0 for w in W])
    bhp = get_well_output(wellSols, "bhp", inj)[0][:, :, 0]
    orat = get_well_output(wellSols, "qOs", prod)[0][:, :, 0]
    wrat = get_well_output(wellSols, "qWs", prod)[0][:, :, 0]
    out = SimpleNamespace(G=G, deck=deck, model=model, schedule=schedule, state0=state,
                          nonlinear=nonlinear, wellSols=wellSols, states=states, report=report,
                          T=T, W=W, inj=inj, prod=prod, bhp=bhp, orat=orat, wrat=wrat)
    if plot:
        _plot_egg(out)
    return out


eggExample = egg_example


def _plot_egg(out: SimpleNamespace) -> None:
    import matplotlib.pyplot as plt
    from pymrst.utils.format_time_range import format_time_range
    W, inj, prod, T = out.W, out.inj, out.prod, out.T
    for data, names, ylabel in (
            (out.bhp/BARSA, [W[i].name for i in inj], "Injector bottom-hole pressure [bar]"),
            (-out.orat/(1/DAY), [W[i].name for i in prod], "Producer oil-rate [m^3/day]"),
            (-out.wrat/(1/DAY), [W[i].name for i in prod], "Producer water-rate [m^3/day]"),
            (np.abs(out.wrat)/np.abs(out.wrat + out.orat), [W[i].name for i in prod],
             "Water cut")):
        plt.figure()
        plt.plot(T, data, linewidth=2)
        plt.legend(names)
        plt.xlabel("Time [years]")
        plt.ylabel(ylabel)
    plt.show()
    # Plot the pressure and water saturation through the simulation
    _animate_egg(out, format_time_range)
    # Launch interactive plotting
    from pymrst.modules.ad_core.plotting.plot_well_sols import plot_well_sols
    from pymrst.modules.ad_blackoil.examples.utils.figures import set_view
    from pymrst.plotting.extensions.plot_toolbar import plot_toolbar
    from pymrst.plotting.plot_well import plot_well
    plot_well_sols(out.wellSols, np.cumsum(out.schedule.step_val))
    toolbar = plot_toolbar(out.G, out.states, show=False)
    plot_well(out.G, [W[i] for i in inj], toolbar.pl, fontsize=12, color="k")
    plot_well(out.G, [W[i] for i in prod], toolbar.pl, fontsize=12, color="r")
    set_view(toolbar.pl, (-50, 70))
    toolbar.show()


def _animate_egg(out: SimpleNamespace, format_time_range) -> None:
    """Pressure (bar) and water saturation side by side, with the wells,
    after each report step."""
    import time

    import pyvista as pv
    from pymrst.modules.ad_blackoil.examples.utils.figures import set_view
    from pymrst.plotting.plot_cell_data import plot_cell_data
    from pymrst.plotting.plot_well import plot_well
    W, inj, prod = out.W, out.inj, out.prod
    plotter = pv.Plotter(shape=(1, 2))
    interactive = not pv.OFF_SCREEN
    if interactive:
        plotter.show(interactive_update=True, auto_close=False)
    for i, st in enumerate(out.states):
        timestr = format_time_range(np.sum(out.schedule.step_val[:i + 1]))
        for k, (data, title, clim) in enumerate((
                (st["pressure"]/BARSA, f"Reservoir pressure after {timestr}", None),
                (np.asarray(st["s"])[:, 0], f"Water saturation after {timestr}", [0, 1]))):
            plotter.subplot(0, k)
            plotter.clear_actors()
            plot_cell_data(out.G, data, plotter=plotter, clim=clim, show=False,
                           show_edges=False)
            plot_well(out.G, [W[j] for j in inj], plotter, fontsize=12, color="k")
            plot_well(out.G, [W[j] for j in prod], plotter, fontsize=12, color="r")
            plotter.add_title(title, font_size=10)
            set_view(plotter, (-50, 70))
        if interactive:
            plotter.update()
            time.sleep(0.01)
    if interactive:
        plotter.show()

@dataclass
class EGGResult:
    """What `eggExample` computes, in the order the script computes it."""

    deck: Dict[str, Any]
    model: Any
    schedule: Any
    state0: Any
    states: List[Any]
    report: Any
    well_sols: List[List[Dict[str, Any]]]
    #: `cumsum(schedule.step.val)` in years.
    T: np.ndarray = field(default_factory=lambda: np.zeros(0))
    #: Injector bottom-hole pressure, `(steps, injectors)`.
    bhp: np.ndarray = field(default_factory=lambda: np.zeros((0, 0)))
    #: Producer oil rate, `(steps, producers)`. Negative, as MRST stores it.
    orat: np.ndarray = field(default_factory=lambda: np.zeros((0, 0)))
    #: Producer water rate, `(steps, producers)`.
    wrat: np.ndarray = field(default_factory=lambda: np.zeros((0, 0)))
    #: `abs(wrat)/abs(wrat + orat)`.
    water_cut: np.ndarray = field(default_factory=lambda: np.zeros((0, 0)))
    injectors: List[str] = field(default_factory=list)
    producers: List[str] = field(default_factory=list)


def run_egg_example(nsteps: Optional[int] = None, *, realization: int = 0,
                    verbose: bool = True, data_root=None) -> EGGResult:
    """Run the Egg benchmark and return MRST's well curves."""
    G, rock, fluid, deck, _ = setup_egg(realization, data_root=data_root)
    state, model, schedule, nonlinear = init_eclipse_problem_ad_ad(
        deck, G=G, timestep_strategy="none")

    if nsteps is not None:
        count = min(int(nsteps), int(schedule.n_steps))
        schedule = Schedule(schedule.step_val[:count],
                            schedule.step_control[:count], schedule.controls)

    nonlinear.verbose = bool(verbose)
    states, report = nonlinear.simulate_schedule(model, state, schedule)
    well_sols = [well_sol_from_state(model, s) for s in states]

    T = np.cumsum(np.asarray(schedule.step_val, dtype=float))/YEAR
    W = list(schedule.controls[0].W)
    inj = [i for i, well in enumerate(W) if int(well.sign) > 0]
    prod = [i for i, well in enumerate(W) if int(well.sign) < 0]

    bhp, injectors = get_well_output_ad(well_sols, "bhp", inj)
    orat, producers = get_well_output_ad(well_sols, "qOs", prod)
    wrat, _ = get_well_output_ad(well_sols, "qWs", prod)
    bhp, orat, wrat = bhp[:, :, 0], orat[:, :, 0], wrat[:, :, 0]

    return EGGResult(
        deck=deck, model=model, schedule=schedule, state0=state,
        states=states, report=report, well_sols=well_sols,
        T=T, bhp=bhp, orat=orat, wrat=wrat,
        water_cut=np.abs(wrat)/np.maximum(np.abs(wrat + orat), 1e-30),
        injectors=injectors, producers=producers,
    )




__all__ = ["EGGResult", "eggExample", "egg_example", "run_egg_example", "well_sol_from_state"]


if __name__ == "__main__":

    def _available() -> bool:
        from pymrst.modules.deckformat.egg.egg_dataset import (
            egg_data_directory,
        )
        try:
            egg_data_directory()
            return True
        except FileNotFoundError:
            return False

    if not _available():
        print("skip (Egg dataset not installed)")
        raise SystemExit(0)

    # Six report steps: 1, 4, 10, 15, 30, 30 days. Enough for the injectors to
    # reach their targets and the producers to draw down, which is what the
    # checks below are about; the full 123 steps take a few minutes and are
    # run from `tests/`.
    RESULT = run_egg_example(6, verbose=False)

    def test_the_schedule_is_the_decks():
        assert np.isclose(RESULT.T[0]*YEAR/DAY, 1.0)
        assert np.isclose(RESULT.T[-1]*YEAR/DAY, 90.0), RESULT.T[-1]*YEAR/DAY
        print("ok test_the_schedule_is_the_decks")

    def test_every_step_converged():
        assert all(step.converged for step in RESULT.report.steps)
        print(f"ok test_every_step_converged "
              f"({sum(s.iterations for s in RESULT.report.steps)} iterations)")

    def test_the_wells_split_eight_to_four_by_their_sign():
        """MRST selects with `find(vertcat(W.sign) > 0)`. The Egg model has
        eight injectors and four producers."""
        assert RESULT.injectors == [f"INJECT{i}" for i in range(1, 9)]
        assert RESULT.producers == [f"PROD{i}" for i in range(1, 5)]
        print("ok test_the_wells_split_eight_to_four_by_their_sign")

    def test_the_injectors_hold_their_declared_rate():
        """79.5 m3/day, from `WCONINJE`. They are rate-controlled and their
        420 bar limit is never reached, so the rate is exactly the target."""
        rates = np.asarray([[row["qWs"] for row in step
                             if row["sign"] > 0]
                            for step in RESULT.well_sols])*DAY
        assert np.allclose(rates, 79.5), rates[-1]
        print("ok test_the_injectors_hold_their_declared_rate")

    def test_the_injectors_stay_under_their_pressure_limit():
        assert np.all(RESULT.bhp <= 420*BARSA + 1.0), RESULT.bhp.max()/BARSA
        assert np.all(RESULT.bhp > 400*BARSA), RESULT.bhp.min()/BARSA
        print(f"ok test_the_injectors_stay_under_their_pressure_limit "
              f"({RESULT.bhp.max()/BARSA:.1f} bar)")

    def test_the_producers_sit_on_their_bottom_hole_target():
        """395 bar, from `WCONPROD`. Nothing else constrains them."""
        bhp = np.asarray([[row["bhp"] for row in step if row["sign"] < 0]
                          for step in RESULT.well_sols])
        assert np.allclose(bhp, 395*BARSA), bhp[-1]/BARSA
        print("ok test_the_producers_sit_on_their_bottom_hole_target")

    def test_the_producers_produce_oil():
        """MRST plots `-orat`: a producer's rate is stored negative, and the
        sign is what tells a producer from an injector in the output."""
        assert np.all(RESULT.orat < 0.0), RESULT.orat[-1]
        rate = -RESULT.orat[-1]*DAY
        assert np.all(rate > 1.0), rate
        print(f"ok test_the_producers_produce_oil "
              f"({rate.min():.0f}-{rate.max():.0f} m3/day)")

    def test_the_water_cut_starts_at_zero():
        """The water-oil contact is at 5000 m, below a reservoir that ends at
        4028 m, so the field starts at connate water and produces none until
        the injected water breaks through. A nonzero cut at day one means the
        initial saturation came from somewhere other than `EQUIL`."""
        assert np.allclose(RESULT.water_cut[0], 0.0), RESULT.water_cut[0]
        print("ok test_the_water_cut_starts_at_zero")

    def test_injected_water_reaches_the_formation():
        """The check that the run is doing anything at all: connate water is
        0.1 everywhere at the start, and the injectors displace oil."""
        s_first = np.asarray(RESULT.states[0]["sW"])
        s_last = np.asarray(RESULT.states[-1]["sW"])
        assert np.isclose(s_first.min(), 0.1, atol=1e-3)
        assert s_last.max() > 0.5, s_last.max()
        print(f"ok test_injected_water_reaches_the_formation "
              f"(max sW {s_last.max():.2f})")

    def test_the_field_stays_within_its_well_pressures():
        p = np.asarray(RESULT.states[-1]["pressure"])/BARSA
        assert p.min() > 380.0 and p.max() < 430.0, (p.min(), p.max())
        print(f"ok test_the_field_stays_within_its_well_pressures "
              f"({p.min():.1f}-{p.max():.1f} bar)")

    def test_a_shortened_run_is_the_same_schedule_truncated():
        short = run_egg_example(2, verbose=False)
        assert np.allclose(short.T, RESULT.T[:2])
        assert np.allclose(short.bhp, RESULT.bhp[:2])
        print("ok test_a_shortened_run_is_the_same_schedule_truncated")

    test_the_schedule_is_the_decks()
    test_every_step_converged()
    test_the_wells_split_eight_to_four_by_their_sign()
    test_the_injectors_hold_their_declared_rate()
    test_the_injectors_stay_under_their_pressure_limit()
    test_the_producers_sit_on_their_bottom_hole_target()
    test_the_producers_produce_oil()
    test_the_water_cut_starts_at_zero()
    test_injected_water_reaches_the_formation()
    test_the_field_stays_within_its_well_pressures()
    test_a_shortened_run_is_the_same_schedule_truncated()
    print("\nAll tests passed!")
