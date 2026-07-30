"""Extract iteration counts and timings from simulation reports."""

from __future__ import annotations

from types import SimpleNamespace
import warnings

import numpy as np


def _get(obj, name, default=None):
    return obj.get(name, default) if isinstance(obj, dict) else getattr(obj, name, default)


def _linear_solver_time(report):
    if not bool(_get(report, "Solved", False)):
        return 0.0
    linear = _get(report, "LinearSolver")
    if linear is None or _get(linear, "SolverTime") is None:
        return 0.0
    return float(_get(linear, "SolverTime")) + float(
        _get(linear, "PostProcessTime", 0.0)
    )


def _getter(kind):
    if kind == "nonlinearIterations":
        return lambda report: float(bool(_get(report, "Solved", False)))
    if kind == "nonlinearSolverTime":
        return lambda report: float(_get(report, "AssemblyTime", 0.0)) + _linear_solver_time(report)
    if kind == "linearIterations":
        def iterations(report):
            if not bool(_get(report, "Solved", False)):
                return 0.0
            linear = _get(report, "LinearSolver")
            values = _get(linear, "Iterations") if linear is not None else None
            return 0.0 if values is None else float(np.asarray(values).ravel()[0])
        return iterations
    if kind == "linearSolverTime":
        return _linear_solver_time
    warnings.warn(
        f"Unknown output type {kind}, switching to nonlinear iterations",
        RuntimeWarning,
        stacklevel=2,
    )
    return _getter("nonlinearIterations")


def _step_data(step, getter, report_level, solver):
    if solver:
        total = 0.0
        local_time = 0.0
        nonlinear = list(_get(step, "NonlinearReport", []))
        for report in nonlinear:
            nested = _get(report, solver)
            nested_output, nested_time = _control_data(
                nested, getter, report_level, None, False
            )
            total += float(np.asarray(nested_output)[0])
            local_time += float(np.sum(nested_time))
        return np.array([
            total,
            total * (not bool(_get(step, "Converged", False))),
            float(len(nonlinear)),
        ]), local_time
    if report_level == "step":
        total = getter(step)
    elif report_level == "nonlinear":
        total = sum(getter(report) for report in _get(step, "NonlinearReport", []))
    else:
        raise ValueError("report_level must be 'nonlinear' or 'step'")
    total = np.atleast_1d(np.asarray(total, dtype=float))
    output = np.column_stack([
        total,
        total * (not bool(_get(step, "Converged", False))),
        np.full(total.size, np.nan),
    ])
    return output.squeeze(), float(_get(step, "LocalTime", 0.0))


def _control_data(control, getter, report_level, solver, ministeps):
    steps = _get(control, "StepReports")
    if steps is None:
        return np.zeros((1, 4)), np.asarray([0.0])
    rows = []
    times = []
    for step in steps:
        output, local_time = _step_data(step, getter, report_level, solver)
        output = np.asarray(output, dtype=float)
        if output.ndim == 1:
            output = output[None, :]
        cut = float(not bool(_get(step, "Converged", False)))
        output = np.column_stack([output, np.full(output.shape[0], cut)])
        rows.append(output)
        times.append(local_time)
    if ministeps:
        return np.vstack(rows), np.asarray(times, dtype=float)
    return np.sum(np.stack(rows), axis=0), np.asarray([sum(times)], dtype=float)


def get_report_output(
    reports,
    *,
    solver=None,
    get=None,
    type="nonlinearIterations",
    report_level="nonlinear",
    ministeps=False,
):
    """Port MRST ``getReportOutput`` for loaded report structures."""
    controls = _get(reports, "ControlstepReports", reports)
    controls = list(controls)
    if ministeps and solver:
        raise ValueError("ministep output for a specific solver is not implemented")
    getter = get if get is not None else _getter(type)
    outputs = []
    times = []
    for control in controls:
        output, local_time = _control_data(
            control, getter, report_level, solver, ministeps
        )
        outputs.append(output)
        times.append(local_time)
    matrix = np.vstack(outputs)
    time = np.concatenate(times)
    result = SimpleNamespace(
        total=matrix[:, 0],
        wasted=matrix[:, 1],
        cuts=matrix[:, 3],
        time=time,
    )
    if solver:
        result.steps = matrix[:, 2]
    return result


getReportOutput = get_report_output

__all__ = ["get_report_output", "getReportOutput"]
