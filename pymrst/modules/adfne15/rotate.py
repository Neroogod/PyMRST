"""Native port of ADFNE15 ``Rotate.m``."""

import numpy as np


def _rotate_points(points, *, x=0.0, y=0.0, z=0.0, angle=0.0,
                   center=None, move=None):
    points = np.asarray(points, dtype=float)
    dimension = points.shape[1]
    if dimension == 2:
        if angle == 0:
            return points.copy()
        pivot = np.mean(points, axis=0) if center is None else np.asarray(center, float)
        radians = np.deg2rad(angle)
        matrix = np.array([[np.cos(radians), np.sin(radians)],
                           [-np.sin(radians), np.cos(radians)]])
        return (points - pivot) @ matrix + pivot
    if dimension != 3:
        raise ValueError("points must be two- or three-dimensional")
    matrix = np.eye(3)
    if z:
        a = np.deg2rad(z); matrix = matrix @ np.array([[np.cos(a), -np.sin(a), 0], [np.sin(a), np.cos(a), 0], [0, 0, 1]])
    if y:
        a = np.deg2rad(y); matrix = matrix @ np.array([[np.cos(a), 0, np.sin(a)], [0, 1, 0], [-np.sin(a), 0, np.cos(a)]])
    if x:
        a = np.deg2rad(x); matrix = matrix @ np.array([[1, 0, 0], [0, np.cos(a), -np.sin(a)], [0, np.sin(a), np.cos(a)]])
    pivot = np.zeros(3) if center is None else np.asarray(center, float)
    transformed = (points - pivot) @ matrix.T + pivot
    target = np.mean(points, axis=0) if move is None else np.asarray(move, float)
    return transformed - np.mean(transformed, axis=0) + target


def rotate(values, *, x=0.0, y=0.0, z=0.0, angle=0.0,
           center=None, move=None):
    """Rotate points, packed lines, or polygons using ADFNE angle order."""
    if isinstance(values, (list, tuple)):
        output = [_rotate_points(value, x=x, y=y, z=z, angle=angle,
                                 center=center, move=move) for value in values]
        return output[0] if len(output) == 1 else output
    values = np.asarray(values, dtype=float)
    columns = values.shape[1]
    if columns in (2, 3):
        return _rotate_points(values, x=x, y=y, z=z, angle=angle,
                              center=center, move=move)
    if columns not in (4, 6):
        raise ValueError("input must contain points or packed line endpoints")
    dimension = columns // 2
    points = values.reshape(-1, dimension)
    # ADFNE forces packed-line rotation about the aggregate point center.
    rotated = _rotate_points(points, x=x, y=y, z=z, angle=angle,
                             center=np.mean(points, axis=0), move=move)
    return rotated.reshape(values.shape)
