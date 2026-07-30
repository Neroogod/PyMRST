"""Stability clipping around a critical point."""

from __future__ import annotations

import numpy as np


def critical_point_chop(
    x0, x_final, xc, epsilon: float = 1e-8,
    chop_type: str = "both", direction: str = "any",
):
    """Port MRST ``criticalPointChop``."""
    if epsilon <= 0:
        raise ValueError("epsilon must be positive")
    x0 = np.asarray(x0, dtype=float)
    out = np.asarray(x_final, dtype=float).copy()
    xc = np.broadcast_to(np.asarray(xc, dtype=float), x0.shape)
    inc = (x0 < xc) & (out > xc)
    dec = (x0 > xc) & (out < xc)
    key = direction.lower()
    if key == "any":
        active = inc | dec
    elif key == "increasing":
        active = inc
    elif key == "decreasing":
        active = dec
    else:
        raise ValueError(f"unknown direction {direction!r}")
    x, before, critical = out[active], x0[active], xc[active]
    increasing, decreasing = inc[active], dec[active]
    key = chop_type.lower()
    if key == "both":
        middle = (before < critical + epsilon) & (before > critical - epsilon)
        lo, hi = increasing & ~middle, increasing & middle
        x[hi] = np.minimum(x[hi], critical[hi] + 1.5 * epsilon)
        x[lo] = critical[lo] - 0.5 * epsilon
        lo, hi = decreasing & middle, decreasing & ~middle
        x[lo] = np.maximum(x[lo], critical[lo] - 1.5 * epsilon)
        x[hi] = critical[hi] + 0.5 * epsilon
    elif key == "above":
        x[increasing] = np.minimum(x[increasing], critical[increasing] + epsilon)
        x[decreasing] = np.maximum(x[decreasing], critical[decreasing] - epsilon)
    elif key == "below":
        mask = increasing & (before < critical - epsilon)
        x[mask] = np.minimum(x[mask], critical[mask] - 0.5 * epsilon)
        mask = decreasing & (before > critical + epsilon)
        x[mask] = np.maximum(x[mask], critical[mask] + 0.5 * epsilon)
    else:
        raise ValueError(f"unknown chop type {chop_type!r}")
    out[active] = x
    return out


criticalPointChop = critical_point_chop

__all__ = ["critical_point_chop", "criticalPointChop"]
