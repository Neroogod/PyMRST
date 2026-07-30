"""Reconstruct global simulation time from reports."""

from __future__ import annotations

import numpy as np


def _get(obj, name, default=None):
    return obj.get(name, default) if isinstance(obj, dict) else getattr(obj, name, default)


def get_simulation_time(states, report):
    """Return cumulative reservoir time for control-step or ministep states."""
    reports = list(report) if isinstance(report, (list, tuple)) else None
    if reports is not None and len(states) == len(reports):
        steps = [float(_get(item, "SimulationTime")) for item in reports]
    elif reports is None and len(states) == 1 and _get(report, "SimulationTime") is not None:
        steps = np.asarray(_get(report, "SimulationTime"), dtype=float).ravel()
    else:
        controls = _get(report, "ControlstepReports", reports or [])
        steps = []
        for control in controls:
            for step in _get(control, "StepReports", []):
                if bool(_get(step, "Converged", False)):
                    steps.append(float(_get(step, "Timestep")))
    return np.cumsum(np.asarray(steps, dtype=float))


getSimulationTime = get_simulation_time

__all__ = ["get_simulation_time", "getSimulationTime"]
