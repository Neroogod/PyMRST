"""MRST ``ad-blackoil/examples/utils/getReportTimings.m``: where a run
spent its wall time, per report step, from the MRST-shaped report that
`simulate_schedule_ad` returns.

`get_report_timings_ad` (with `Timing` and `report_ministeps`) is the
earlier version for the legacy `NonLinearSolver.simulate_schedule` report,
kept for `blackoil_tutorial_mex_acceleration`. Its notes follow.

Where a run spent its time (MRST
``ad-blackoil/examples/utils/getReportTimings.m``).

MRST returns one record per report step with five fields - `Total`,
`Assembly`, `LinearSolve`, `LinearSolvePrep` and `NumberOfAssemblies` - and
`blackoilTutorialMexAcceleration` bars them against each other to show where
acceleration actually lands.

This file previously returned **the ministep durations**: simulated time, in
seconds of reservoir time, not wall time. Same name, same shape, different
quantity - and plotted as a timing breakdown it would have shown a perfectly
flat "cost" per step, because the schedule's steps are equal.

The breakdown is only recoverable if it is recorded while the run happens, so
`StepReport` now carries `wall_seconds` and `assembly_seconds`, and each
`LinearSolveDiagnostic` carries its own `wall_seconds`. `Other` - what MRST
computes as `Total - (Assembly + LinearSolve + LinearSolvePrep)` - is what is
left: state updates, convergence checks, the timestep selector.

`LinearSolvePrep` is zero here. It is the time MRST's external solvers spend
building a preconditioner before the first Krylov iteration; a SciPy direct
solve has no such phase, and reporting a made-up split would misattribute the
factorisation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Union

import numpy as np


def _f(obj: Any, name: str) -> Any:
    return obj[name] if isinstance(obj, dict) else getattr(obj, name)


def _isfield(obj: Any, name: str) -> bool:
    return name in obj if isinstance(obj, dict) else hasattr(obj, name)


def _make_timing(total: float, assembly: float, linear: float, linear_prep: float,
                 its: float, asm: float) -> Dict[str, float]:
    return {"Total": total, "Assembly": assembly, "LinearSolve": linear,
            "LinearSolvePrep": linear_prep, "NumberOfAssemblies": asm, "Iterations": its}


def get_report_timings(report: Any, skipConverged: bool = False,
                       total: bool = False) -> Union[List[Dict[str, float]], Dict[str, float]]:
    """MRST's `getReportTimings`: per report step, its wall time and the time
    spent assembling and in the linear solver (with preparation), the number
    of assemblies and Newton iterations; their sums with `total`.

    `report` is `simulateScheduleAD`'s report or its `ControlstepReports`.
    With `skipConverged` the assembly that only confirmed convergence is not
    counted.
    """
    if isinstance(report, dict) and "ControlstepReports" in report:
        report = report["ControlstepReports"]
    nan = float("nan")
    timings = [_make_timing(nan, nan, nan, nan, nan, nan) for _ in report]
    for step_no, rep in enumerate(report):
        assembly = lsolve = lsolve_prep = 0.0
        n_assemblies = 0
        for ministep in _f(rep, "StepReports"):
            nonlinear = _f(ministep, "NonlinearReport")
            n_assemblies += len(nonlinear)
            for r in nonlinear:
                conv = _f(r, "Converged")
                if skipConverged and conv:
                    continue
                assembly += _f(r, "AssemblyTime")
                ls = _f(r, "LinearSolver")
                if not conv:
                    prep = 0.0
                    if _isfield(ls, "PreparationTime"):
                        prep += _f(ls, "PreparationTime")
                        prep += _f(ls, "PostProcessTime")
                        lsolve += _f(ls, "LinearSolutionTime")
                    elif _isfield(ls, "SolverTime"):
                        lsolve += _f(ls, "SolverTime")
                    lsolve_prep += prep
        timings[step_no] = _make_timing(_f(rep, "WallTime"), assembly, lsolve, lsolve_prep,
                                        _f(rep, "Iterations"), n_assemblies)
    if total:
        return {name: sum(t[name] for t in timings) for name in timings[0]}
    return timings


getReportTimings = get_report_timings


# ---------------------------------------------------------------------------
# The earlier version, kept for `blackoil_tutorial_mex_acceleration`, whose
# solver reports are not MRST's.

@dataclass
class Timing:
    """One report step's wall-clock breakdown, in seconds."""

    total: float = 0.0
    assembly: float = 0.0
    linear_solve: float = 0.0
    linear_solve_prep: float = 0.0
    n_assemblies: int = 0

    @property
    def other(self) -> float:
        """What `Total` is not accounted for by the three named parts."""
        return self.total - (self.assembly + self.linear_solve
                             + self.linear_solve_prep)


def get_report_timings_ad(report: Any, *, skip_converged: bool = False,
                       total: bool = False):
    """Per-step timings, or their sum when ``total`` is set."""
    steps = list(getattr(report, "steps", None) or [report])
    timings: List[Timing] = []
    for step in steps:
        solves = list(getattr(step, "linear_solves", ()) or ())
        if skip_converged and getattr(step, "converged", False):
            # MRST's `skipConverged` drops the *converged* assembly - the one
            # that only confirms the answer and solves nothing.
            solves = solves[:-1] if solves else solves
        timings.append(Timing(
            total=float(getattr(step, "wall_seconds", 0.0)),
            assembly=float(getattr(step, "assembly_seconds", 0.0)),
            linear_solve=float(sum(getattr(s, "wall_seconds", 0.0)
                                   for s in solves)),
            linear_solve_prep=0.0,
            n_assemblies=len(getattr(step, "residuals", ()) or ()),
        ))
    if not total:
        return timings
    return Timing(
        total=float(sum(t.total for t in timings)),
        assembly=float(sum(t.assembly for t in timings)),
        linear_solve=float(sum(t.linear_solve for t in timings)),
        linear_solve_prep=float(sum(t.linear_solve_prep for t in timings)),
        n_assemblies=int(sum(t.n_assemblies for t in timings)),
    )


def report_ministeps(report: Any) -> np.ndarray:
    """The *simulated* step lengths - what this module used to return.

    Kept under an honest name because it is a useful thing, just not a timing.
    """
    steps = getattr(report, "steps", None) or [report]
    out: List[float] = []
    for step in steps:
        mini = getattr(step, "ministeps", None)
        out.extend(mini if mini else [float(getattr(step, "dt", 0.0))])
    return np.asarray(out, dtype=float)



__all__ = ["Timing", "getReportTimings", "get_report_timings",
           "get_report_timings_ad", "report_ministeps"]


if __name__ == "__main__":
    def _synthetic_report():
        ls = {"PreparationTime": 0.5, "PostProcessTime": 0.25, "LinearSolutionTime": 2.0}
        nonlinear = [{"Converged": False, "AssemblyTime": 1.0, "LinearSolver": ls},
                     {"Converged": False, "AssemblyTime": 1.5, "LinearSolver": {"SolverTime": 3.0}},
                     {"Converged": True, "AssemblyTime": 0.75, "LinearSolver": {}}]
        step = {"WallTime": 10.0, "Iterations": 2,
                "StepReports": [{"NonlinearReport": nonlinear}]}
        return {"ControlstepReports": [step, dict(step, WallTime=20.0)]}

    def test_mrst_fields_per_report_step():
        t = get_report_timings(_synthetic_report())
        assert t[0] == {"Total": 10.0, "Assembly": 3.25, "LinearSolve": 5.0,
                        "LinearSolvePrep": 0.75, "NumberOfAssemblies": 3, "Iterations": 2}
        assert t[1]["Total"] == 20.0
        print("ok test_mrst_fields_per_report_step")

    def test_skip_converged_and_total():
        t = get_report_timings(_synthetic_report(), skipConverged=True, total=True)
        assert t["Assembly"] == 5.0 and t["NumberOfAssemblies"] == 6 and t["Total"] == 30.0
        print("ok test_skip_converged_and_total")

    test_mrst_fields_per_report_step()
    test_skip_converged_and_total()

    from pymrst.gridprocessing import cart_grid, compute_geometry
    from pymrst.modules.ad_blackoil.models import GenericBlackOilModelAD
    from pymrst.modules.ad_core import State
    from pymrst.modules.ad_core.simulators.nonlinear_solver import (
        NonLinearSolver,
    )
    from pymrst.modules.ad_core.utils.schedule import simple_schedule
    from pymrst.modules.ad_props import init_simple_adi_fluid
    from pymrst.params import add_well, make_rock
    from pymrst.utils.units import BARSA, CENTI, DARCY, DAY, MILLI, POISE

    def _run(n=8, steps=3):
        g = compute_geometry(cart_grid([n, n, 2], [100.0, 100.0, 10.0]))
        rock = make_rock(g, 100*MILLI*DARCY, 0.3)
        fluid = init_simple_adi_fluid(
            mu=np.array([1.0, 5.0])*CENTI*POISE, rho=[1000.0, 800.0],
            n=[2, 2], c=[1e-6/BARSA, 1e-6/BARSA], p_ref=200*BARSA)
        W = add_well(None, g, rock, np.array([0]), well_type="rate",
                     val=1e-4, compi=np.array([1.0, 0.0]), sign=1, name="I")
        W = add_well(W, g, rock, np.array([g.cells.num - 1]), well_type="bhp",
                     val=150*BARSA, compi=np.array([0.0, 1.0]), sign=-1,
                     name="P")
        model = GenericBlackOilModelAD(g, rock, fluid, W)
        state = State(pressure=np.full(g.cells.num, 200*BARSA),
                      sW=np.full(g.cells.num, 0.2),
                      qWs=np.array([1e-4, 0.0]), qOs=np.array([0.0, 0.0]),
                      bhp=np.array([210*BARSA, 150*BARSA]))
        solver = NonLinearSolver()
        solver.verbose = False
        return solver.simulate_schedule(
            model, state, simple_schedule(np.full(steps, 10.0*DAY), W=W))[1]

    REPORT = _run()

    def test_it_returns_one_record_per_report_step():
        timings = get_report_timings_ad(REPORT)
        assert len(timings) == len(REPORT.steps) == 3
        print("ok test_it_returns_one_record_per_report_step")

    def test_the_numbers_are_wall_time_not_simulated_time():
        """What this module used to return. A ten-day step takes 864000
        seconds of *reservoir* time and a fraction of a second of wall time;
        plotted as a cost breakdown the former is flat and meaningless."""
        timings = get_report_timings_ad(REPORT)
        assert all(0.0 < t.total < 60.0 for t in timings), [t.total for t in timings]
        assert np.allclose(report_ministeps(REPORT), 10.0*DAY)
        print(f"ok test_the_numbers_are_wall_time_not_simulated_time "
              f"({timings[0].total:.3f} s vs {10.0*DAY:.0f} s simulated)")

    def test_assembly_and_solve_are_both_measured_and_both_smaller_than_total():
        timings = get_report_timings_ad(REPORT)
        for t in timings:
            assert t.assembly > 0.0, t
            assert t.linear_solve > 0.0, t
            assert t.assembly + t.linear_solve <= t.total + 1e-9, t
        print("ok test_assembly_and_solve_are_both_measured_and_both_smaller_than_total")

    def test_other_is_the_unaccounted_remainder():
        """MRST's fifth bar: `Total` minus the three named parts. A negative
        one would mean a part was counted twice."""
        for t in get_report_timings_ad(REPORT):
            assert t.other >= -1e-9, t
            assert np.isclose(t.total, t.assembly + t.linear_solve
                              + t.linear_solve_prep + t.other)
        print("ok test_other_is_the_unaccounted_remainder")

    def test_total_sums_the_steps():
        parts = get_report_timings_ad(REPORT)
        whole = get_report_timings_ad(REPORT, total=True)
        assert np.isclose(whole.total, sum(t.total for t in parts))
        assert np.isclose(whole.assembly, sum(t.assembly for t in parts))
        print("ok test_total_sums_the_steps")

    def test_skip_converged_drops_the_confirming_assembly():
        """The last assembly of a converged step solves nothing - it only
        confirms the answer. MRST's `skipConverged` excludes it."""
        full = get_report_timings_ad(REPORT, total=True)
        trimmed = get_report_timings_ad(REPORT, skip_converged=True, total=True)
        assert trimmed.linear_solve <= full.linear_solve
        print("ok test_skip_converged_drops_the_confirming_assembly")

    def test_the_number_of_assemblies_is_recorded():
        timings = get_report_timings_ad(REPORT)
        assert all(t.n_assemblies > 0 for t in timings)
        print("ok test_the_number_of_assemblies_is_recorded")

    test_it_returns_one_record_per_report_step()
    test_the_numbers_are_wall_time_not_simulated_time()
    test_assembly_and_solve_are_both_measured_and_both_smaller_than_total()
    test_other_is_the_unaccounted_remainder()
    test_total_sums_the_steps()
    test_skip_converged_drops_the_confirming_assembly()
    test_the_number_of_assemblies_is_recorded()
    print("\nAll tests passed!")
