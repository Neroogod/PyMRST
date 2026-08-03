"""Finite planar fracture-intersection NNCs in 3D."""

from collections.abc import Sequence

import numpy as np

from pymrst.modules.hfm import EDFMGrid, PlanarFracture, Projection3D
from pymrst.modules.shale.data import ShaleNNC


def _clip_line_to_fracture(
    point: np.ndarray,
    direction: np.ndarray,
    fracture: PlanarFracture,
    tolerance: float,
) -> tuple[float, float] | None:
    _, axis1, axis2 = fracture.frame()
    relative = point - np.asarray(fracture.center, dtype=float)
    polygon = fracture.local_polygon()
    origin = np.array([relative @ axis1, relative @ axis2], dtype=float)
    slope = np.array([direction @ axis1, direction @ axis2], dtype=float)
    lower, upper = -np.inf, np.inf
    for index, edge_start in enumerate(polygon):
        edge = polygon[(index + 1) % len(polygon)] - edge_start
        intercept = edge[0] * (origin[1] - edge_start[1]) - edge[1] * (origin[0] - edge_start[0])
        rate = edge[0] * slope[1] - edge[1] * slope[0]
        if abs(rate) <= tolerance:
            if intercept < -tolerance:
                return None
            continue
        boundary = -intercept / rate
        if rate > 0:
            lower = max(lower, boundary)
        else:
            upper = min(upper, boundary)
        if upper < lower - tolerance:
            return None
    return lower, upper


def _rectangle_intersection_segment(
    first: PlanarFracture,
    second: PlanarFracture,
    tolerance: float,
) -> tuple[np.ndarray, np.ndarray] | None:
    n1, _, _ = first.frame()
    n2, _, _ = second.frame()
    cross = np.cross(n1, n2)
    norm2 = float(cross @ cross)
    if norm2 <= tolerance**2:
        return None
    direction = cross / np.sqrt(norm2)
    d1 = float(n1 @ np.asarray(first.center, dtype=float))
    d2 = float(n2 @ np.asarray(second.center, dtype=float))
    point = (d1 * np.cross(n2, cross) + d2 * np.cross(cross, n1)) / norm2
    interval1 = _clip_line_to_fracture(point, direction, first, tolerance)
    interval2 = _clip_line_to_fracture(point, direction, second, tolerance)
    if interval1 is None or interval2 is None:
        return None
    lower = max(interval1[0], interval2[0])
    upper = min(interval1[1], interval2[1])
    if upper - lower <= tolerance:
        return None
    return point + lower * direction, point + upper * direction


def fracture_fracture_shale_nncs_3d(
    edfm: EDFMGrid,
    fractures: Sequence[PlanarFracture],
    projections: Sequence[Projection3D],
    *,
    aperture: float | Sequence[float],
    tolerance: float = 1e-10,
) -> ShaleNNC:
    """Create cross-plane NNCs for intersecting finite rectangular fractures."""
    if len(fractures) != len(projections):
        raise ValueError("fractures and projections must have the same length")
    apertures = np.asarray(aperture, dtype=float).reshape(-1)
    if apertures.size == 1:
        apertures = np.full(len(fractures), apertures.item())
    if apertures.size != len(fractures) or np.any(apertures <= 0):
        raise ValueError("aperture must be positive and scalar or per-fracture")
    offsets = np.cumsum([edfm.n_matrix] + [p.matrix_cells.size for p in projections])
    cells: list[tuple[int, int]] = []
    transmissibility: list[float] = []
    for first in range(len(fractures) - 1):
        for second in range(first + 1, len(fractures)):
            segment = _rectangle_intersection_segment(
                fractures[first], fractures[second], tolerance
            )
            if segment is None:
                continue
            start, end = segment
            direction = end - start
            length = float(np.linalg.norm(direction))
            direction = direction / length
            p1, p2 = projections[first], projections[second]
            if not p1.fracture_cells.size or not p2.fracture_cells.size:
                continue
            intervals1 = _cell_line_intervals(start, direction, length, p1, fractures[first], tolerance)
            intervals2 = _cell_line_intervals(start, direction, length, p2, fractures[second], tolerance)
            for local1, lower1, upper1 in intervals1:
                for local2, lower2, upper2 in intervals2:
                    lower = max(lower1, lower2)
                    upper = min(upper1, upper2)
                    sublength = upper - lower
                    if sublength <= tolerance:
                        continue
                    midpoint = start + 0.5 * (lower + upper) * direction
                    cell1 = int(offsets[first] + local1)
                    cell2 = int(offsets[second] + local2)
                    d1 = max(float(np.linalg.norm(p1.centroids[local1] - midpoint)), apertures[first] / 2)
                    d2 = max(float(np.linalg.norm(p2.centroids[local2] - midpoint)), apertures[second] / 2)
                    k1, k2 = float(edfm.perm[cell1]), float(edfm.perm[cell2])
                    resistance = (
                        d1 / (k1 * apertures[first] * sublength)
                        + d2 / (k2 * apertures[second] * sublength)
                    )
                    cells.append((cell1, cell2))
                    transmissibility.append(1.0 / resistance)
    count = len(cells)
    return ShaleNNC(
        cells=np.asarray(cells, dtype=np.int64).reshape(-1, 2),
        transmissibility=np.asarray(transmissibility),
        connection_type=("fracfrac",) * count,
        area=np.zeros(count),
        normal=np.zeros((count, 3)),
        plane_point=np.zeros((count, 3)),
    )


def _cell_line_intervals(start, direction, length, projection, fracture, tolerance):
    """Return line intervals covered by every fracture cell polygon."""
    _, axis1, axis2 = fracture.frame()
    intervals = []
    for local_cell, polygon3d in enumerate(projection.polygons):
        center = np.asarray(fracture.center, dtype=float)
        polygon = (polygon3d - center) @ np.column_stack([axis1, axis2])
        origin3d = start - center
        origin = np.array([origin3d @ axis1, origin3d @ axis2])
        slope = np.array([direction @ axis1, direction @ axis2])
        lower, upper = 0.0, length
        for index, edge_start in enumerate(polygon):
            edge = polygon[(index + 1) % len(polygon)] - edge_start
            intercept = edge[0] * (origin[1] - edge_start[1]) - edge[1] * (origin[0] - edge_start[0])
            rate = edge[0] * slope[1] - edge[1] * slope[0]
            if abs(rate) <= tolerance:
                if intercept < -tolerance:
                    lower, upper = 1.0, 0.0
                    break
                continue
            boundary = -intercept / rate
            if rate > 0:
                lower = max(lower, boundary)
            else:
                upper = min(upper, boundary)
            if upper <= lower + tolerance:
                break
        if upper > lower + tolerance:
            intervals.append((local_cell, max(lower, 0.0), min(upper, length)))
    return intervals


__all__ = ["fracture_fracture_shale_nncs_3d"]
