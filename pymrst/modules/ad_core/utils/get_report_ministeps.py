"""Extract converged ministep lengths from simulation reports."""

from __future__ import annotations

import numpy as np


def _get(obj, name, default=None):
    return obj.get(name, default) if isinstance(obj, dict) else getattr(obj, name, default)


def get_report_ministeps(report):
    """Port MRST ``getReportMinisteps``."""
    controls = report if isinstance(report, (list, tuple)) else _get(
        report, "ControlstepReports", _get(report, "controlstep_reports", [])
    )
    timesteps = []
    for control in controls:
        steps = _get(control, "StepReports", _get(control, "step_reports", []))
        for step in steps:
            if bool(_get(step, "Converged", _get(step, "converged", False))):
                timesteps.append(float(_get(step, "Timestep", _get(step, "timestep"))))
    return np.asarray(timesteps)


getReportMinisteps = get_report_ministeps

__all__ = ["get_report_ministeps", "getReportMinisteps"]
