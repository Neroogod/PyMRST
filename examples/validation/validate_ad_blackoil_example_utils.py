"""Validate MRST's example utilities `getReportTimings` and `surfWithOutline`.

The reference is `dump_ad_blackoil_example_utils_reference.m` (MRST 2024a).
`getReportTimings` is fed the same fixed report on both sides (wall times
cannot be compared between runs): per step, with `skipConverged`, with
`total`, and on `ControlstepReports` directly - exactly. `surfWithOutline`:
the surface drawn, and its four black outlines in MATLAB's order, width and
colour - exactly.
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.io import loadmat

from pymrst.modules.ad_blackoil.examples.utils.get_report_timings import getReportTimings
from pymrst.modules.ad_blackoil.examples.utils.surf_with_outline import surfWithOutline

HERE = Path(__file__).resolve().parent
FIELDS = ("Total", "Assembly", "LinearSolve", "LinearSolvePrep", "NumberOfAssemblies",
          "Iterations")


def report(ok: bool, text: str) -> bool:
    print(f"{'ok  ' if ok else 'FAIL'} {text}")
    return ok


def _report():
    ls = {"PreparationTime": 0.5, "PostProcessTime": 0.25, "LinearSolutionTime": 2.0}
    nonlinear = [{"Converged": False, "AssemblyTime": 1.0, "LinearSolver": ls},
                 {"Converged": False, "AssemblyTime": 1.5, "LinearSolver": {"SolverTime": 3.0}},
                 {"Converged": True, "AssemblyTime": 0.75, "LinearSolver": {}}]
    step = {"WallTime": 10.0, "Iterations": 2, "StepReports": [{"NonlinearReport": nonlinear}]}
    return {"ControlstepReports": [step, dict(step, WallTime=20.0)]}


def _pack(timings) -> np.ndarray:
    rows = timings if isinstance(timings, list) else [timings]
    return np.array([[t[f] for f in FIELDS] for t in rows], dtype=float).squeeze()


def main() -> int:
    ref = loadmat(HERE/"ref_ad_blackoil_example_utils.mat", squeeze_me=True)
    rep = _report()
    ok = True
    for key, timings in (("timings", getReportTimings(rep)),
                         ("timings_skip", getReportTimings(rep, skipConverged=True)),
                         ("timings_total", getReportTimings(rep, total=True)),
                         ("timings_cell", getReportTimings(rep["ControlstepReports"]))):
        ok &= report(np.array_equal(_pack(timings), ref[key]), f"getReportTimings: {key}")
    plt.figure()
    surfWithOutline(ref["X"], ref["Y"], ref["Z"])
    ax = plt.gca()
    surface = ax.collections
    ok &= report(len(surface) == 1, "surfWithOutline draws one surface")
    lines = ax.get_lines()
    ok &= report(len(lines) == int(ref["nlines"]), f"surfWithOutline: {len(lines)} outlines")
    outline = np.vstack([np.column_stack(l.get_data_3d()) for l in lines])
    ok &= report(np.array_equal(outline, ref["outline"]),
                 "surfWithOutline: outline points, in MATLAB's order")
    ok &= report(all(matplotlib.colors.to_rgb(l.get_color()) == tuple(ref["color"].astype(float))
                     and l.get_linewidth() == float(ref["width"]) for l in lines),
                 "surfWithOutline: outlines black, width 2")
    plt.close("all")
    print("\nThe example utilities match MRST" if ok
          else "\nThe example utilities do NOT match MRST")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
