"""Projection matrix/fracture NNC discovery (`pMatFracNNCs3D`)."""

from collections.abc import Sequence
from itertools import permutations

import numpy as np

from pymrst.modules.hfm import EDFMGrid, PlanarFracture, Projection3D
from pymrst.modules.shale.data import ProjectedNNC


def _cell_faces(grid, cell: int) -> np.ndarray:
    return np.asarray(
        grid.cells.faces[grid.cells.facePos[cell]:grid.cells.facePos[cell + 1]],
        dtype=np.int64,
    )


def _axis_faces(grid, cell: int, tolerance: float) -> list[tuple[int, int]]:
    """Return three opposing face pairs ordered by best global-axis match."""
    center = np.asarray(grid.cells.centroids[cell, :3])
    faces = _cell_faces(grid, cell)
    offsets = np.asarray(grid.faces.centroids[faces, :3]) - center
    if faces.size != 6:
        raise ValueError("pMatFracNNCs3D requires six-face hexahedral cells")
    norms = np.linalg.norm(offsets, axis=1)
    if np.any(norms <= tolerance):
        raise ValueError("cell face centroid coincides with its cell centroid")
    directions = offsets / norms[:, None]
    unused = set(range(6))
    raw_pairs = []
    while unused:
        first = min(unused)
        candidates = np.asarray(sorted(unused - {first}), dtype=np.int64)
        second = int(candidates[np.argmin(directions[candidates] @ directions[first])])
        if directions[first] @ directions[second] > -0.5:
            raise ValueError("hexahedral faces cannot be paired as opposites")
        raw_pairs.append((first, second))
        unused.remove(first)
        unused.remove(second)
    pair_axes = np.asarray([
        directions[second] - directions[first] for first, second in raw_pairs
    ])
    pair_axes /= np.linalg.norm(pair_axes, axis=1)[:, None]
    assignment = max(
        permutations(range(3)),
        key=lambda order: sum(abs(pair_axes[order[axis], axis]) for axis in range(3)),
    )
    ordered = []
    for axis, pair_index in enumerate(assignment):
        first, second = raw_pairs[pair_index]
        if offsets[first, axis] <= offsets[second, axis]:
            ordered.append((int(faces[first]), int(faces[second])))
        else:
            ordered.append((int(faces[second]), int(faces[first])))
    return ordered


def _other_neighbor(grid, face: int, host: int) -> int | None:
    neighbors = np.asarray(grid.faces.neighbors[face], dtype=np.int64)
    other = neighbors[neighbors != host]
    if other.size != 1 or other[0] < 0:
        return None
    return int(other[0])


def projected_matrix_fracture_nncs_3d(
    edfm: EDFMGrid,
    fractures: Sequence[PlanarFracture],
    projections: Sequence[Projection3D],
    *,
    tolerance: float = 1e-10,
    default_face: str = "bottom",
) -> ProjectedNNC:
    """Construct the three directional pEDFM projections per interior cell.

    The face closest to the fracture-cell centroid is selected along each axis.
    Exact ties use the negative face, matching MRST's deterministic default-face
    branch. Duplicate matrix pairs accumulate projected area conservatively and
    are capped by their geometric face area.
    """
    if edfm.g.griddim != 3:
        raise ValueError("pMatFracNNCs3D requires a 3D matrix grid")
    if len(fractures) != len(projections):
        raise ValueError("fractures and projections must have the same length")
    default_face = str(default_face).lower()
    if default_face not in {"left", "front", "bottom"}:
        raise ValueError("default_face must be 'left', 'front', or 'bottom'")
    mf_cells: list[tuple[int, int]] = []
    mm_pairs: list[tuple[int, int]] = []
    trans: list[float] = []
    area_out: list[float] = []
    directions: list[int] = []
    face_ids: list[int] = []
    projected_polygons: list[np.ndarray] = []
    frac_offset = edfm.n_matrix
    for fracture, projection in zip(fractures, projections):
        normal, _, _ = fracture.frame()
        for local, host in enumerate(projection.matrix_cells):
            host = int(host)
            frac_cell = frac_offset + local
            frac_center = np.asarray(projection.centroids[local], dtype=float)
            frac_area = float(projection.areas[local])
            for axis, (negative, positive) in enumerate(
                _axis_faces(edfm.g, host, tolerance)
            ):
                dneg = np.linalg.norm(edfm.g.faces.centroids[negative, :3] - frac_center)
                dpos = np.linalg.norm(edfm.g.faces.centroids[positive, :3] - frac_center)
                face = positive if dpos + tolerance < dneg else negative
                projected = _other_neighbor(edfm.g, face, host)
                if projected is None:
                    continue
                face_normal = np.asarray(edfm.g.faces.normals[face, :3], dtype=float)
                face_normal /= np.linalg.norm(face_normal)
                projected_area = frac_area * abs(float(face_normal @ normal))
                distance = np.linalg.norm(
                    edfm.g.cells.centroids[projected, :3] - frac_center
                )
                if distance <= tolerance:
                    raise ValueError("projected matrix/fracture distance is zero")
                k_matrix = float(edfm.perm[projected])
                k_frac = float(edfm.perm[frac_cell])
                if k_matrix <= 0 or k_frac <= 0:
                    raise ValueError("pEDFM permeability must be positive")
                harmonic = 2.0 / (1.0 / k_matrix + 1.0 / k_frac)
                pair = tuple(map(int, edfm.g.faces.neighbors[face]))
                mf_cells.append((projected, frac_cell))
                mm_pairs.append(pair)
                area_out.append(projected_area)
                trans.append(harmonic * projected_area / distance)
                directions.append(axis + 1)
                face_ids.append(face)
                polygon = np.asarray(projection.polygons[local], dtype=float)
                projected_polygons.append(np.delete(polygon, axis, axis=1))
        frac_offset += projection.matrix_cells.size

    # MRST retains one projected NNC per matrix pair and replaces repeated
    # projection area with a union area. Without rasterization, coincident
    # contributions are accumulated and bounded by the exact host-face area.
    unique: dict[tuple[int, int], int] = {}
    keep_mf: list[tuple[int, int]] = []
    keep_mm: list[tuple[int, int]] = []
    keep_t: list[float] = []
    keep_area: list[float] = []
    keep_dir: list[int] = []
    keep_face: list[int] = []
    keep_polygons: list[list[np.ndarray]] = []
    for mf, mm, value, projected_area, direction, face, polygon in zip(
        mf_cells, mm_pairs, trans, area_out, directions, face_ids, projected_polygons
    ):
        key = tuple(sorted(mm))
        if key not in unique:
            unique[key] = len(keep_mm)
            keep_mf.append(mf)
            keep_mm.append(mm)
            keep_t.append(value)
            keep_area.append(projected_area)
            keep_dir.append(direction)
            keep_face.append(face)
            keep_polygons.append([polygon])
        else:
            index = unique[key]
            keep_polygons[index].append(polygon)
            combined = min(
                _convex_polygon_union_area(keep_polygons[index], tolerance),
                float(edfm.g.faces.areas[keep_face[index]]),
            )
            keep_area[index] = combined
    return ProjectedNNC(
        matrix_fracture_cells=np.asarray(keep_mf, dtype=np.int64).reshape(-1, 2),
        matrix_matrix_neighbors=np.asarray(keep_mm, dtype=np.int64).reshape(-1, 2),
        transmissibility=np.asarray(keep_t),
        area=np.asarray(keep_area),
        direction=np.asarray(keep_dir, dtype=np.int64),
        connection_type=("fracpmat",) * len(keep_mf),
    )


def _convex_polygon_union_area(polygons, tolerance=1e-10):
    """Exact union area for convex 2D polygons by vertical slab integration."""
    polygons = [np.asarray(p, dtype=float).reshape(-1, 2) for p in polygons]
    events = [float(x) for polygon in polygons for x in polygon[:, 0]]
    for first in range(len(polygons) - 1):
        for second in range(first + 1, len(polygons)):
            for a1, a2 in _edges(polygons[first]):
                for b1, b2 in _edges(polygons[second]):
                    point = _segment_intersection(a1, a2, b1, b2, tolerance)
                    if point is not None:
                        events.append(float(point[0]))
    events = np.unique(np.round(np.asarray(events) / tolerance) * tolerance)
    area = 0.0
    for left, right in zip(events[:-1], events[1:]):
        width = right - left
        if width <= tolerance:
            continue
        x = 0.5 * (left + right)
        intervals = []
        for polygon in polygons:
            interval = _vertical_interval(polygon, x, tolerance)
            if interval is not None:
                intervals.append(interval)
        area += width * _interval_union_length(intervals, tolerance)
    return float(area)


def _edges(polygon):
    return zip(polygon, np.roll(polygon, -1, axis=0))


def _segment_intersection(a1, a2, b1, b2, tolerance):
    r, s = a2 - a1, b2 - b1
    denominator = r[0] * s[1] - r[1] * s[0]
    if abs(denominator) <= tolerance:
        return None
    delta = b1 - a1
    t = (delta[0] * s[1] - delta[1] * s[0]) / denominator
    u = (delta[0] * r[1] - delta[1] * r[0]) / denominator
    if -tolerance <= t <= 1 + tolerance and -tolerance <= u <= 1 + tolerance:
        return a1 + t * r
    return None


def _vertical_interval(polygon, x, tolerance):
    values = []
    for first, second in _edges(polygon):
        dx = second[0] - first[0]
        if abs(dx) <= tolerance:
            continue
        fraction = (x - first[0]) / dx
        if -tolerance <= fraction <= 1 + tolerance:
            values.append(first[1] + fraction * (second[1] - first[1]))
    if len(values) < 2:
        return None
    return min(values), max(values)


def _interval_union_length(intervals, tolerance):
    if not intervals:
        return 0.0
    intervals = sorted(intervals)
    total = 0.0
    lower, upper = intervals[0]
    for start, end in intervals[1:]:
        if start <= upper + tolerance:
            upper = max(upper, end)
        else:
            total += upper - lower
            lower, upper = start, end
    return total + upper - lower


__all__ = ["projected_matrix_fracture_nncs_3d"]
