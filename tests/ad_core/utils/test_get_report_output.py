from types import SimpleNamespace

import numpy as np

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
    return [
        SimpleNamespace(StepReports=[failed, success]),
        SimpleNamespace(StepReports=[second]),
    ]


def test_report_output_iterations_waste_cuts_and_time():
    output = get_report_output(_reports())
    assert np.array_equal(output.total, [2.0, 1.0])
    assert np.array_equal(output.wasted, [1.0, 0.0])
    assert np.array_equal(output.cuts, [1.0, 0.0])
    assert np.array_equal(output.time, [6.0, 3.0])

    linear = get_report_output(_reports(), type="linearIterations")
    assert np.array_equal(linear.total, [7.0, 2.0])
    assert np.array_equal(linear.wasted, [3.0, 0.0])


def test_report_output_ministeps_and_solver_times():
    mini = get_report_output(
        SimpleNamespace(ControlstepReports=_reports()),
        type="nonlinearSolverTime",
        ministeps=True,
    )
    assert np.allclose(mini.total, [3.7, 1.85, 1.1])
    assert np.allclose(mini.wasted, [3.7, 0.0, 0.0])
    assert np.array_equal(mini.cuts, [1.0, 0.0, 0.0])
    assert np.array_equal(mini.time, [4.0, 2.0, 3.0])
