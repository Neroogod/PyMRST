"""Port of ADFNE15 ``Intersect.m`` for 2-D lines and convex 3-D polygons."""

from dataclasses import dataclass
import numpy as np

from .bbox import bounding_box
from .plane import plane
from .polyline import polyline
from .unique import unique_rows
from .cluster import cluster


@dataclass(frozen=True, slots=True)
class IntersectionResult:
    geometry: np.ndarray
    pairs: np.ndarray
    labels: np.ndarray


def intersect(items, tolerance=1e-14):
    if isinstance(items, (list, tuple)):
        polygons = tuple(np.asarray(value, dtype=float) for value in items)
        candidates = bounding_box(polygons, intersections=True)
        geometry, pairs = [], []
        for pair in candidates:
            line = _polygon_intersection(polygons[pair[0]], polygons[pair[1]], tolerance)
            if line is not None:
                geometry.append(line); pairs.append(pair)
        width = 6
    else:
        lines = np.atleast_2d(np.asarray(items, dtype=float))
        if lines.shape[1] != 4:
            raise ValueError("Intersect requires n-by-4 lines or 3-D polygons")
        candidates = bounding_box(lines, intersections=True)
        geometry, pairs = [], []
        for pair in candidates:
            point = _segment_intersection(lines[pair[0]], lines[pair[1]], tolerance)
            if point is not None:
                geometry.append(point); pairs.append(pair)
        width = 2
    pairs = np.asarray(pairs, dtype=np.int64).reshape(-1, 2)
    count = len(items)
    groups = cluster(pairs) if len(pairs) else []
    labels = np.zeros(count, dtype=np.int64)
    for number, group in enumerate(groups, start=1): labels[group] = number
    labels[labels == 0] = -np.arange(1, np.count_nonzero(labels == 0)+1)
    return IntersectionResult(np.asarray(geometry, dtype=float).reshape(-1, width), pairs, labels)


def _segment_intersection(first, second, tolerance):
    p, r = first[:2], first[2:]-first[:2]
    q, s = second[:2], second[2:]-second[:2]
    cross = lambda a, b: a[0]*b[1]-a[1]*b[0]
    denominator = cross(r, s)
    if abs(denominator) <= tolerance:
        return None
    t, u = cross(q-p, s)/denominator, cross(q-p, r)/denominator
    if -tolerance <= t <= 1+tolerance and -tolerance <= u <= 1+tolerance:
        return p+t*r
    return None


def _polygon_intersection(first, second, tolerance):
    points = []
    for polygon, other in ((first, second), (second, first)):
        definition = plane(other); normal = np.cross(definition[3:6], definition[6:9])
        for edge in polyline(polygon):
            start, delta = edge[:3], edge[3:]-edge[:3]
            denominator = normal @ delta
            if abs(denominator) <= tolerance: continue
            parameter = normal @ (definition[:3]-start)/denominator
            if -tolerance <= parameter <= 1+tolerance:
                point = start+parameter*delta
                if _inside_convex(point, other, normal, tolerance): points.append(point)
    if len(points) < 2: return None
    points = unique_rows(points, max(tolerance, 1e-12))
    if len(points) < 2: return None
    distance = np.linalg.norm(points[:, None, :]-points[None, :, :], axis=2)
    i, j = np.unravel_index(np.argmax(distance), distance.shape)
    endpoints = np.vstack((points[i], points[j]))
    order = np.lexsort(tuple(endpoints[:, column]
                             for column in range(endpoints.shape[1]-1, -1, -1)))
    return np.r_[endpoints[order[0]], endpoints[order[1]]]


def _inside_convex(point, polygon, normal, tolerance):
    signs = []
    for index in range(len(polygon)):
        edge = polygon[(index+1) % len(polygon)]-polygon[index]
        signs.append(np.dot(np.cross(edge, point-polygon[index]), normal))
    signs = np.asarray(signs)
    return bool(np.all(signs >= -tolerance) or np.all(signs <= tolerance))
