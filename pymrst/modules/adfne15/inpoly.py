"""Native port of ADFNE15 ``Inpoly.m``."""

import numpy as np

from .plane import plane
from .polyline import polyline
from .project import project
from .snap import snap


def _inside_2d(points, polygon):
    x, y = points[:, 0], points[:, 1]
    x0, y0 = polygon[:, 0], polygon[:, 1]
    x1, y1 = np.roll(x0, -1), np.roll(y0, -1)
    crossing = ((y0[None, :] > y[:, None]) != (y1[None, :] > y[:, None])) & (
        x[:, None] < (x1-x0)[None, :]*(y[:, None]-y0[None, :]) /
        (y1-y0 + np.finfo(float).eps)[None, :] + x0[None, :])
    return np.count_nonzero(crossing, axis=1) % 2 == 1


def inpoly(points, polygon, tolerance=1e-14):
    """Test whether 2-D/3-D points are inside or on a finite polygon."""
    points = np.atleast_2d(np.asarray(points, dtype=float))
    polygon = np.asarray(polygon, dtype=float)
    if points.shape[1] != polygon.shape[1] or points.shape[1] not in (2, 3):
        raise ValueError("points and polygon must have matching dimension")
    if points.shape[1] == 3:
        definition = plane(polygon)
        normal = np.cross(definition[3:6], definition[6:9])
        normal /= np.linalg.norm(normal)
        close_plane = np.abs((points-definition[:3]) @ normal) <= tolerance
    else:
        definition = None
        close_plane = np.ones(len(points), dtype=bool)
    result = np.zeros(len(points), dtype=bool)
    candidates = np.flatnonzero(close_plane)
    if not len(candidates):
        return result
    on_edge = np.any(snap(points[candidates], polyline(polygon), 0,
                          distances_only=True) <= tolerance, axis=1)
    result[candidates[on_edge]] = True
    remaining = candidates[~on_edge]
    if len(remaining):
        if definition is None:
            flat_points, flat_polygon = points[remaining], polygon
        else:
            values = project(np.vstack((points[remaining], polygon)), definition)
            flat_points, flat_polygon = values[:len(remaining)], values[len(remaining):]
        result[remaining] = _inside_2d(flat_points, flat_polygon)
    return result
