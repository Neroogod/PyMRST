"""Port of MRST ``examples/utils/getReportTimings.m``."""

from __future__ import annotations

import numpy as np


def get_report_timings(report) -> np.ndarray:
    """Return ministep durations from a PyMRST SimulationReport/StepReport."""
    steps = getattr(report, "steps", [report])
    out = []
    for step in steps:
        mini = getattr(step, "ministeps", None)
        if mini:
            out.extend(mini)
        else:
            out.append(getattr(step, "dt", 0.0))
    return np.asarray(out, dtype=float)


__all__ = ["get_report_timings"]
