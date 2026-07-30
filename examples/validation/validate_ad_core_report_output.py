"""Validate ``getReportOutput`` against MRST."""

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import scipy.io

from pymrst.modules.ad_core.utils import get_report_output


def _nonlinear(solved, assembly, iterations, solver_time, post=0.0):
    return SimpleNamespace(
        Solved=solved,
        AssemblyTime=assembly,
        LinearSolver=SimpleNamespace(
            Iterations=np.array([iterations]),
            SolverTime=solver_time,
            PostProcessTime=post,
        ),
    )


def _reports():
    failed = SimpleNamespace(
        Converged=False,
        LocalTime=4.0,
        NonlinearReport=[
            _nonlinear(True, 1.0, 3, 2.0, 0.5),
            _nonlinear(False, 0.2, 9, 8.0),
        ],
    )
    success = SimpleNamespace(
        Converged=True,
        LocalTime=2.0,
        NonlinearReport=[
            _nonlinear(True, 0.5, 4, 1.0, 0.25),
            _nonlinear(False, 0.1, 8, 7.0),
        ],
    )
    second = SimpleNamespace(
        Converged=True,
        LocalTime=3.0,
        NonlinearReport=[_nonlinear(True, 0.25, 2, 0.75, 0.1)],
    )
    return SimpleNamespace(ControlstepReports=[
        SimpleNamespace(StepReports=[failed, success]),
        SimpleNamespace(StepReports=[second]),
    ])


def main() -> int:
    ref = scipy.io.loadmat(
        Path(__file__).parent / "ref_ad_core_report_output.mat",
        squeeze_me=True,
    )
    default = get_report_output(_reports())
    linear = get_report_output(_reports(), type="linearIterations")
    mini = get_report_output(
        _reports(), type="nonlinearSolverTime", ministeps=True
    )
    linear_time = get_report_output(_reports(), type="linearSolverTime")
    checks = {
        "default total": (default.total, ref["default_total"]),
        "default wasted": (default.wasted, ref["default_wasted"]),
        "default cuts": (default.cuts, ref["default_cuts"]),
        "default time": (default.time, ref["default_time"]),
        "linear total": (linear.total, ref["linear_total"]),
        "linear wasted": (linear.wasted, ref["linear_wasted"]),
        "ministep total": (mini.total, ref["mini_total"]),
        "ministep wasted": (mini.wasted, ref["mini_wasted"]),
        "ministep cuts": (mini.cuts, ref["mini_cuts"]),
        "ministep time": (mini.time, ref["mini_time"]),
        "linear time total": (
            linear_time.total,
            ref["linear_time_total"],
        ),
    }
    failed = 0
    for name, (actual, expected) in checks.items():
        actual = np.asarray(actual, dtype=float).squeeze()
        expected = np.asarray(expected, dtype=float).squeeze()
        error = float(np.max(np.abs(actual - expected)))
        ok = actual.shape == expected.shape and np.allclose(
            actual, expected, rtol=1e-12, atol=1e-12
        )
        print(f"{'ok  ' if ok else 'FAIL'} {name}: max abs diff {error:.3e}")
        failed += not ok
    return int(failed != 0)


if __name__ == "__main__":
    raise SystemExit(main())
