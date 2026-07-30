"""MRST ``twister`` coordinate deformation."""

import numpy as np


def twister(value, amplitude=0.03):
    """Twist a coordinate array or a grid in place-compatible form."""
    if hasattr(value, "nodes") and hasattr(value.nodes, "coords"):
        value.nodes.coords = twister(value.nodes.coords, amplitude)
        return value
    points = np.asarray(value, dtype=float)
    if points.ndim != 2 or points.shape[1] not in (2, 3):
        raise TypeError("twister expects an (n, 2/3) coordinate array or Grid")
    lo, span = points.min(axis=0), np.ptp(points, axis=0)
    safe_span = np.where(span == 0, 1.0, span)
    xi = (points - lo) / safe_span

    def f(a, b):
        return amplitude * np.sin(np.pi * a) * np.sin(3 * (-np.pi / 2 + np.pi * b))

    out = xi.copy()
    out[:, 0] += f(xi[:, 0], xi[:, 1])
    out[:, 1] -= f(xi[:, 1], xi[:, 0])
    if points.shape[1] == 3:
        out[:, 1] += f(xi[:, 1], xi[:, 2])
        out[:, 2] -= f(xi[:, 2], xi[:, 1])
        out[:, 0] += f(xi[:, 0], xi[:, 2])
        out[:, 2] -= f(xi[:, 2], xi[:, 0])
    return lo + out * span
