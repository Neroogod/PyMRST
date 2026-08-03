"""Native port of ADFNE15 ``Convert.m`` conversion modes."""

import numpy as np

from .center import center
from .plane import plane
from .project import project


def as_cell(value, *, return_was_cell=False):
    was_cell = isinstance(value, (list, tuple))
    if isinstance(value, dict):
        output = [part for pair in value.items() for part in pair]
    elif was_cell:
        output = list(value)
    else:
        output = [value]
    return (output, was_cell) if return_was_cell else output


def clock(seconds):
    hours = int(np.floor(seconds/3600))
    minutes = int(np.floor((seconds-hours*3600)/60))
    remainder = np.remainder(seconds, 3600)-minutes*60
    return f"{hours:02d}:{minutes:02d}:{remainder:05.2f}"


def normalized(values):
    values = np.asarray(values, dtype=float)
    one_dimensional = values.ndim == 1
    rows = np.atleast_2d(values)
    output = rows/np.linalg.norm(rows, axis=1, keepdims=True)
    return output.reshape(-1) if one_dimensional else output


def polygon_triangles(polygons):
    collection = list(polygons) if isinstance(polygons, (list, tuple)) else [polygons]
    output = []
    for polygon in collection:
        polygon = np.asarray(polygon, dtype=float)
        centroid = np.asarray(center([polygon])).reshape(-1)
        output.append([np.vstack((centroid, polygon[i], polygon[(i+1) % len(polygon)]))
                       for i in range(len(polygon))])
    return output


def as_struct(*items):
    if len(items) % 2:
        raise ValueError("structure conversion requires key/value pairs")
    return {str(items[i]): items[i+1] for i in range(0, len(items), 2)}


def to_2d(points, plane_definition=None, centered=False):
    points = np.asarray(points, dtype=float)
    definition = plane(points) if plane_definition is None else plane_definition
    original_center = np.asarray(center(points)).reshape(-1) if centered else None
    output = project(points, definition)
    if centered:
        output -= np.asarray(center(output)).reshape(-1)-original_center[:2]
    return output


def to_3d(values):
    def add_z(value):
        value = np.asarray(value, dtype=float)
        return np.column_stack((value, np.zeros(len(value))))
    candidate = np.asarray(values)
    if (isinstance(values, (list, tuple)) and
            (candidate.dtype == object or candidate.ndim != 2)):
        return [add_z(value) for value in values]
    values = np.asarray(values, dtype=float)
    if values.shape[1] == 4:
        return np.hstack((add_z(values[:, :2]), add_z(values[:, 2:])))
    if values.shape[1] == 2:
        return add_z(values)
    return values.copy()


def convert(mode, *values, **options):
    name = str(mode).lower()
    if name == "cell": return as_cell(values[0], **options)
    if name == "clock": return clock(values[0])
    if name == "deg": return np.rad2deg(values[0])
    if name == "rad": return np.deg2rad(values[0])
    if name == "norm": return normalized(values[0])
    if name == "tri": return polygon_triangles(values[0])
    if name == "struct": return as_struct(*values)
    if name == "2d": return to_2d(*values, **options)
    if name == "3d": return to_3d(values[0])
    raise ValueError(f"unknown conversion mode: {mode}")
