"""Create a schedule from converged report ministeps."""

from __future__ import annotations

import numpy as np

from pymrst.modules.ad_core.utils.schedule import Schedule


def _get(obj, name, default=None):
    return obj.get(name, default) if isinstance(obj, dict) else getattr(obj, name, default)


def convert_report_to_schedule(report, schedule):
    """Replace schedule steps by the converged ministeps in ``report``."""
    controls_reports = _get(report, "ControlstepReports", report)
    timesteps = []
    controls = []
    counts = []
    original_controls = np.asarray(schedule.step_control, dtype=np.int64)
    for index, control_report in enumerate(controls_reports):
        converged = [
            step
            for step in _get(control_report, "StepReports", [])
            if bool(_get(step, "Converged", False))
        ]
        timesteps.extend(float(_get(step, "Timestep")) for step in converged)
        controls.extend([int(original_controls[index])] * len(converged))
        counts.append(len(converged))
    converted = Schedule(
        step_val=np.asarray(timesteps, dtype=float),
        step_control=np.asarray(controls, dtype=np.int64),
        controls=list(schedule.controls),
    )
    return converted, np.asarray(timesteps, dtype=float), np.asarray(counts, dtype=float)


convertReportToSchedule = convert_report_to_schedule

__all__ = ["convert_report_to_schedule", "convertReportToSchedule"]
