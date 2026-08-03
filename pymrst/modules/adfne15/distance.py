"""Native port of ADFNE15 ``Distance.m``."""

import numpy as np

from .plane import plane
from .polyline import polyline
from .project import project
from .snap import snap


def distance(points, objects):
    """Return minimum distances from points to lines or finite polygons."""
    points = np.atleast_2d(np.asarray(points, dtype=float))
    if not isinstance(objects, (list, tuple)):
        return np.min(snap(points, objects, 0, distances_only=True), axis=1)
    values = np.empty((len(points), len(objects)), dtype=float)
    for column, item in enumerate(objects):
        polygon = np.asarray(item, dtype=float)
        if polygon.shape[1] == 3:
            definition = plane(polygon)
            normal = np.cross(definition[3:6], definition[6:9])
            normal /= np.linalg.norm(normal)
            plane_distance = np.abs((points - definition[:3]) @ normal)
            projected = project(np.vstack((points, polygon)), definition)
            flat_points, flat_polygon = projected[:len(points)], projected[len(points):]
        else:
            plane_distance = np.zeros(len(points))
            flat_points, flat_polygon = points, polygon
        x, y = flat_points[:, 0], flat_points[:, 1]
        x0, y0 = flat_polygon[:, 0], flat_polygon[:, 1]
        x1, y1 = np.roll(x0, -1), np.roll(y0, -1)
        crosses = ((y0[None, :] > y[:, None]) != (y1[None, :] > y[:, None])) & (
            x[:, None] < (x1-x0)[None, :]*(y[:, None]-y0[None, :]) /
            (y1-y0 + np.finfo(float).eps)[None, :] + x0[None, :])
        inside = np.count_nonzero(crosses, axis=1) % 2 == 1
        edge_distance = np.min(snap(points, polyline(polygon), 0,
                                    distances_only=True), axis=1)
        values[:, column] = np.where(inside, plane_distance, edge_distance)
    return np.min(values, axis=1)
