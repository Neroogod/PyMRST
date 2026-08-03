"""Port of ADFNE15 ``Sort.m`` topological point ordering."""

import numpy as np


def sort_points(points, direction=None):
    points = np.asarray(points, dtype=float)
    if len(points) <= 1: return points.copy()
    if direction is None or np.any(np.isnan(direction)):
        distance = np.linalg.norm(points[:, None, :]-points[None, :, :], axis=2)
        first, second = np.unravel_index(np.argmax(distance), distance.shape)
        origin, vector = points[first], points[second]-points[first]
    else:
        direction = np.asarray(direction, dtype=float); d = direction.size//2
        origin, vector = direction[:d], direction[d:]-direction[:d]
    parameter = (points-origin) @ vector/max(vector @ vector, np.finfo(float).eps)
    return points[np.argsort(parameter, kind="stable")]
