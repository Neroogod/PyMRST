"""Extract signed well-summary matrices from timestep well solutions."""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np


def _get(row, name):
    return row[name] if isinstance(row, dict) else getattr(row, name)


def well_sol_to_vector(well_sols: Sequence[Sequence[Any]]):
    """Port MRST ``wellSolToVector`` returning ``qWs, qOs, qGs, bhp``."""
    steps = [list(step) for step in well_sols]
    if not steps:
        empty = np.zeros((0, 0))
        return empty.copy(), empty.copy(), empty.copy(), empty.copy()
    count = len(steps[0])
    if any(len(step) != count for step in steps):
        raise ValueError("wellSolToVector does not support variable well counts")

    def matrix(field):
        return np.asarray([[_get(row, field) for row in step] for step in steps], dtype=float)

    sign = matrix("sign")
    bhp = matrix("bhp")
    return sign * matrix("qWs"), sign * matrix("qOs"), sign * matrix("qGs"), bhp


wellSolToVector = well_sol_to_vector

__all__ = ["well_sol_to_vector", "wellSolToVector"]
