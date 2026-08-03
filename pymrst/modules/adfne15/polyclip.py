"""Native port of ADFNE15 ``PolyClip.m``."""

import numpy as np


def _clip_halfspace(polygon, boundary):
    boundary = np.asarray(boundary, dtype=float)
    origin = boundary[0]
    # This matches Plane(boundary, true): the vertex ordering defines the
    # outward normal and the non-positive half-space is retained.
    normal = np.cross(boundary[1]-origin, np.mean(boundary, axis=0)-origin)
    norm = np.linalg.norm(normal)
    if norm <= 1e-15:
        raise ValueError("clipping boundary is degenerate")
    normal /= norm
    polygon = np.asarray(polygon, dtype=float)
    if not len(polygon):
        return polygon.reshape(0, 3)
    closed = (polygon if np.array_equal(polygon[0], polygon[-1])
              else np.vstack((polygon, polygon[0])))
    values = (closed-origin) @ normal
    inside = values <= 0
    if not np.any(inside):
        return np.empty((0, 3))
    if np.all(inside):
        return closed
    transitions = np.flatnonzero(inside != np.roll(inside, -1))
    if len(transitions) != 2:
        raise ValueError("polygon must cross a clipping plane exactly twice")
    intersections = []
    for index in transitions:
        following = (index+1) % len(closed)
        fraction = values[index]/(values[index]-values[following])
        intersections.append(closed[index] + fraction*(closed[following]-closed[index]))
    intersections = np.asarray(intersections)
    first, second = transitions
    if inside[0]:
        output = np.vstack((closed[:first+1], intersections,
                            closed[second+1:]))
    else:
        output = np.vstack((intersections[0], closed[first+1:second+1],
                            intersections[1]))
    if len(output) > 1 and np.array_equal(output[0], output[-1]):
        output = output[:-1]
    return output


def polyclip(polygons, boundaries, *, return_mask=False):
    """Clip 3-D polygons by oriented closed boundary half-spaces."""
    polygon_list = list(polygons) if isinstance(polygons, (list, tuple)) else [polygons]
    boundary_list = list(boundaries) if isinstance(boundaries, (list, tuple)) else [boundaries]
    output = []
    mask = np.zeros(len(polygon_list), dtype=bool)
    for index, polygon in enumerate(polygon_list):
        clipped = np.asarray(polygon, dtype=float)
        for boundary in boundary_list:
            clipped = _clip_halfspace(clipped, boundary)
            if not len(clipped):
                break
        if len(clipped) >= 3:
            output.append(clipped)
            mask[index] = True
    return (output, mask) if return_mask else output
