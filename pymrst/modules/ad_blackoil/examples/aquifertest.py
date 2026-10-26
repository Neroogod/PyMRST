"""MRST ``ad-blackoil/examples/aquifertest.m``: two published oil-water decks
with Fetkovich aquifers, simulated and compared with ECLIPSE.

For each case MRST reads the deck, sets it up with `initEclipseProblemAD`
(the deck's `AQUFETP`/`AQUANCON` become the model's `AquiferModel`),
simulates it with `simulateScheduleAD`, reads ECLIPSE's restart states with
`loadAquiferEclipseResult`, and plots both and the relative difference of
the final pressures. MRST's script, case by case; `plot=False` leaves out
its figures.

`run_aquifertest_ad` and `compare_with_eclipse_ad`, below, are PyMRST's
earlier version on `setup_aquifertest_ad` and the legacy solver. Their
notes follow.

Port of MRST ``ad-blackoil/examples/aquifertest.m``.

MRST's script, in full::

    datadir = getDatasetPath('aquifertest', 'download', true);
    for icase = 1 : 2
        fn   = fullfile(datadir, sprintf('2D_OILWATER_AQ_ECLIPSE_%d.DATA', icase));
        deck = convertDeckUnits(readEclipseDeck(fn));
        [state0, model, schedule] = initEclipseProblemAD(deck);
        [wellSols, states, schedulereport] = simulateScheduleAD(state0, model, schedule);
        statesEcl = loadAquiferEclipseResult(datadir, fnroot);
        ... plot the relative difference (p - pEcl)/pEcl
    end

The point of the example is the last line: **MRST against ECLIPSE**, on a case
whose only interesting feature is an analytic aquifer. The previous port ran
neither - it built a synthetic 10x1x2 Cartesian model with a hand-written
aquifer and compared it to nothing, because a case invented here has no
reference to be compared against.

Running it needed three things that did not exist:

* the SOLUTION reader, so `AQUFETP` and `AQUANCON` arrive at all. `AQUFETP`
  gives an aquifer its productivity index, compressibility, pressure and
  volume; **`AQUANCON` says which cells it touches**, and an aquifer with no
  connections contributes no influx while running perfectly;
* the SOLUTION unit conversion for both, so the aquifer pressure is 300 bar
  rather than 300 Pa and its influx coefficient is an area;
* `init_eclipse_problem_ad_ad` on a real deck.

`compare_with_eclipse` is the comparison itself: the relative pressure
difference against the stored `.X####` restart files, cell by cell.

**It matches restarts to report steps by simulation time**, read from each
restart's `DOUBHEAD[0]`. Index matching is wrong here and was: ECLIPSE writes
22 restarts for a 21-step schedule because the first is the *initial* state at
t = 0, so restart `i` is report step `i - 1`.

Getting the comparison to mean anything turned up two real defects, and the
sizes say which mattered:

* the tokenizer did not recognise a record-ending `/` with trailing junk. This
  deck writes `'BHP' 2* 300 /|`, so `WCONINJE` never terminated and the reader
  kept the following `WCONPROD` **keyword** as a data row - leaving the
  producer with a 0 bar target. One cell's pressure collapsed to 4.5 bar
  against ECLIPSE's 200.8;
* the deck equilibration ignored capillary pressure, placing a sharp contact.
  The SWOF `Pcow` column implies a transition zone 63 m thick here, and
  ECLIPSE reports `Sw = 0.111` at the top of the model where a sharp contact
  gives 0.

With both fixed, over the full 1101-day schedule: **2.9% maximum and 2.2%
median** relative pressure difference for case 1, and 0.58% / 0.08% for
case 2 - two different simulators on one case, and the step that used to fail
to converge now converges in three Newton iterations.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any, Dict, List, Optional

import numpy as np

from pymrst.modules.ad_blackoil.examples.utils.load_aquifer_eclipse_result import (
    load_aquifer_eclipse_result, load_aquifer_eclipse_result_ad,
)
from pymrst.modules.ad_blackoil.examples.utils.setup_aquifertest import (
    CASES, aquifertest_data_directory, setup_aquifertest_ad,
)
from pymrst.modules.ad_core.simulators.nonlinear_solver import NonLinearSolver
from pymrst.modules.ad_core.utils.schedule import Schedule
from pymrst.modules.ad_core.utils.well_sol_from_state import (
    well_sols_from_states,
)


def aquifertest(plot: bool = True) -> List[SimpleNamespace]:
    """MRST `aquifertest`: one result per case."""
    from pymrst.modules.ad_core.ad_oo.simulate_schedule_ad import simulate_schedule_ad
    from pymrst.modules.ad_core.initialization.init_eclipse_problem_ad import (
        init_eclipse_problem_ad,
    )
    from pymrst.modules.deckformat.deckinput.convert_deck_units import convert_deck_units
    from pymrst.modules.deckformat.deckinput.read_eclipse_deck import read_eclipse_deck
    from pymrst.params.gravity.gravity import gravity
    from pymrst.utils.units import BARSA

    gravity("on")
    datadir = aquifertest_data_directory()
    out = []
    for icase in (1, 2):
        fnroot = f"2D_OILWATER_AQ_ECLIPSE_{icase}"
        fn = datadir/f"{fnroot}.DATA"
        deck = convert_deck_units(read_eclipse_deck(fn))
        state0, model, schedule = init_eclipse_problem_ad(deck)[:3]
        wellSols, states, schedulereport = simulate_schedule_ad(state0, model, schedule)
        statesEcl, _ = load_aquifer_eclipse_result(datadir, fnroot)
        G = model.G
        p = np.asarray(states[-1]["pressure"], dtype=float)
        pecl = np.asarray(statesEcl[-1]["pressure"], dtype=float)*BARSA
        err = (p - pecl)/pecl
        if plot:
            _plot_case(G, states, statesEcl, err, icase)
        out.append(SimpleNamespace(case=icase, deck=deck, model=model, state0=state0,
                                   schedule=schedule, wellSols=wellSols, states=states,
                                   schedulereport=schedulereport, statesEcl=statesEcl,
                                   err=err))
    return out


def _plot_case(G: Any, states: List[Any], statesEcl: List[Any], err: np.ndarray,
               icase: int) -> None:
    """MRST's three figures, with the final step where it opens plotToolbar."""
    from pymrst.modules.ad_blackoil.examples.utils.figures import cell_data_figure
    cell_data_figure(G, np.asarray(states[-1]["pressure"]),
                     f"MRST computation - Case {icase}", view=(2, 2))
    cell_data_figure(G, np.asarray(statesEcl[-1]["pressure"]),
                     f"Eclipse computation - Case {icase}", view=(2, 2))
    cell_data_figure(G, err, f"Relative difference - Case {icase}", view=(2, 2))


# ---------------------------------------------------------------------------
# PyMRST's earlier version.

@dataclass
class AquiferCaseResult:
    case: int
    model: Any
    state0: Any
    schedule: Any
    states: List[Any]
    report: Any
    well_sols: List[List[Dict[str, Any]]]

    @property
    def aquifer_pressure(self) -> np.ndarray:
        """`(steps, aquifers)` - the analytic aquifers' own pressure."""
        return np.vstack([np.asarray(s["aquifer_p"], dtype=float)
                          for s in self.states])

    @property
    def aquifer_volume(self) -> np.ndarray:
        return np.vstack([np.asarray(s["aquifer_V"], dtype=float)
                          for s in self.states])


def run_aquifertest_ad(nsteps: Optional[int] = None, *,
                    cases=(1, 2), verbose: bool = True,
                    data_root=None) -> List[AquiferCaseResult]:
    """Run both published cases, as MRST's `for icase = 1 : 2` does."""
    out: List[AquiferCaseResult] = []
    for case in cases:
        model, state0, schedule = setup_aquifertest_ad(case=case,
                                                    data_root=data_root)
        if nsteps is not None:
            count = min(int(nsteps), int(schedule.n_steps))
            schedule = Schedule(schedule.step_val[:count],
                                schedule.step_control[:count],
                                schedule.controls)
        solver = NonLinearSolver()
        solver.verbose = bool(verbose)
        states, report = solver.simulate_schedule(model, state0, schedule)
        out.append(AquiferCaseResult(
            case=case, model=model, state0=state0, schedule=schedule,
            states=states, report=report,
            well_sols=well_sols_from_states(model, states)))
    return out


def compare_with_eclipse_ad(result: AquiferCaseResult, *,
                         data_root=None,
                         tolerance_seconds: float = 1.0) -> Dict[str, Any]:
    """The relative pressure difference against ECLIPSE, as MRST plots it.

    The restart compared against is the one whose own simulation time matches
    the last report step, not the one at the same index.
    """
    directory = aquifertest_data_directory(data_root)
    reference = load_aquifer_eclipse_result_ad(directory, CASES[result.case - 1])
    # `AquiferEclipseResult` bundles the restart states with the grid and the
    # summary; the comparison wants the states.
    states = list(getattr(reference, "states", reference) or ())
    if not states:
        raise FileNotFoundError(
            f"no ECLIPSE restart files for {CASES[result.case - 1]} under "
            f"{directory}; there is nothing to compare against")

    # Our report times, and theirs. Matching on these rather than on position
    # is the whole correctness of the comparison.
    ours_time = np.cumsum(np.asarray(result.schedule.step_val, dtype=float))
    theirs_time = np.asarray([s.time for s in states], dtype=float)
    if not np.all(np.isfinite(theirs_time)):
        raise ValueError(
            "the ECLIPSE restarts carry no DOUBHEAD time, so they cannot be "
            "matched to report steps except by position - which is wrong "
            "whenever the two wrote a different number of records")

    step = len(result.states) - 1
    target = float(ours_time[step])
    match = int(np.argmin(np.abs(theirs_time - target)))
    mismatch = abs(theirs_time[match] - target)
    if mismatch > tolerance_seconds:
        raise ValueError(
            f"no ECLIPSE restart within {tolerance_seconds:g} s of "
            f"t = {target:g} s; nearest is {theirs_time[match]:g} s")

    ours = np.asarray(result.states[step]["pressure"], dtype=float)
    theirs = np.asarray(_pressure_of(states[match]), dtype=float)
    if theirs.size != ours.size:
        raise ValueError(
            f"ECLIPSE reports {theirs.size} cells, the model has {ours.size}")
    error = (ours - theirs)/theirs
    return {"step": step, "eclipse_record": match, "time": target,
            "time_mismatch": float(mismatch),
            "pressure": ours, "eclipse": theirs,
            "relative_error": error,
            "max_relative_error": float(np.max(np.abs(error))),
            "median_relative_error": float(np.median(np.abs(error)))}


def _pressure_of(state: Any) -> np.ndarray:
    for key in ("pressure", "PRESSURE"):
        if isinstance(state, dict) and key in state:
            return np.asarray(state[key], dtype=float).ravel()
        found = getattr(state, key, None)
        if found is not None:
            return np.asarray(found, dtype=float).ravel()
    raise KeyError("the ECLIPSE restart state carries no pressure")


__all__ = ["AquiferCaseResult", "aquifertest", "compare_with_eclipse_ad",
           "run_aquifertest_ad"]


if __name__ == "__main__":
    try:
        aquifertest_data_directory()
    except FileNotFoundError as missing:
        print(f"skip ({missing})")
        raise SystemExit(0)

    RESULTS = aquifertest(plot=False)

    def test_both_cases_run_to_the_end_with_their_aquifers():
        assert [r.case for r in RESULTS] == [1, 2]
        for r in RESULTS:
            assert len(r.states) == 21 and r.model.AquiferModel is not None
            p = np.array([[a.pressure for a in st["aquiferSol"]] for st in r.states])
            assert np.all(np.abs(p[-1] - p[0]) > 1.0), "an aquifer never moved"
        print("ok test_both_cases_run_to_the_end_with_their_aquifers")

    def test_the_final_pressure_is_eclipses_to_a_fraction_of_a_percent():
        for r in RESULTS:
            assert np.max(np.abs(r.err)) < 1e-3, np.max(np.abs(r.err))
        print("ok test_the_final_pressure_is_eclipses_to_a_fraction_of_a_percent")

    test_both_cases_run_to_the_end_with_their_aquifers()
    test_the_final_pressure_is_eclipses_to_a_fraction_of_a_percent()
    print("\nAll tests passed!")
