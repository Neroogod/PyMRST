"""Port of ADFNE15 ``Group.m``."""

import numpy as np


def group(intersections, pairs, count):
    points = [[] for _ in range(count)]; neighbours = [[] for _ in range(count)]
    for geometry, (first, second) in zip(np.asarray(intersections), np.asarray(pairs, dtype=int)):
        neighbours[first].append(second); neighbours[second].append(first)
        points[first].append(geometry); points[second].append(geometry)
    width = np.asarray(intersections).shape[1] if np.asarray(intersections).ndim == 2 else 0
    return ([np.asarray(value, dtype=float).reshape(-1, width) for value in points],
            [np.asarray(value, dtype=np.int64) for value in neighbours])
