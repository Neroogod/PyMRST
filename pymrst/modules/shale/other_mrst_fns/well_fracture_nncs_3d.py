"""Piecewise-linear well/fracture intersections in 3D."""

from collections.abc import Sequence

import numpy as np

from pymrst.modules.hfm import PlanarFracture, Projection3D
from pymrst.modules.shale.data import WellFractureConnections


def _segment_rectangle_intersection(
    start: np.ndarray,
    end: np.ndarray,
    fracture: PlanarFracture,
    tolerance: float,
) -> np.ndarray | None:
    normal, axis1, axis2 = fracture.frame()
    direction = end - start
    denominator = float(normal @ direction)
    if abs(denominator) < tolerance:
        return None
    parameter = float(normal @ (np.asarray(fracture.center) - start)) / denominator
    if parameter < -tolerance or parameter > 1.0 + tolerance:
        return None
    point = start + np.clip(parameter, 0.0, 1.0) * direction
    relative = point - np.asarray(fracture.center)
    if abs(float(relative @ axis1)) > fracture.half_length1 + tolerance:
        return None
    if abs(float(relative @ axis2)) > fracture.half_length2 + tolerance:
        return None
    return point


def well_fracture_nncs_3d(
    well_paths: Sequence[np.ndarray],
    fractures: Sequence[PlanarFracture],
    projections: Sequence[Projection3D],
    *,
    matrix_cell_count: int,
    tolerance: float = 1e-10,
) -> WellFractureConnections:
    """Locate well/plane intersections and map them to global fracture cells."""
    if len(fractures) != len(projections):
        raise ValueError("fractures and projections must have equal length")
    offsets = np.cumsum(
        [matrix_cell_count] + [projection.matrix_cells.size for projection in projections]
    )
    rows: list[tuple[int, int, int, np.ndarray, int, int]] = []
    seen: set[tuple[int, int, tuple[int, int, int]]] = set()
    for well_index, raw_path in enumerate(well_paths):
        path = np.asarray(raw_path, dtype=float)
        if path.ndim != 2 or path.shape[1] != 3 or path.shape[0] < 2:
            raise ValueError("each well path must have shape (n, 3), n >= 2")
        for segment in range(path.shape[0] - 1):
            for fracture_index, (fracture, projection) in enumerate(
                zip(fractures, projections)
            ):
                point = _segment_rectangle_intersection(
                    path[segment], path[segment + 1], fracture, tolerance
                )
                if point is None or not projection.fracture_cells.size:
                    continue
                key = (
                    well_index,
                    fracture_index,
                    tuple(np.rint(point / max(tolerance, 1e-12)).astype(np.int64)),
                )
                if key in seen:
                    continue
                seen.add(key)
                local = int(
                    np.argmin(np.linalg.norm(projection.centroids - point, axis=1))
                )
                rows.append((
                    well_index,
                    segment,
                    fracture_index,
                    point,
                    int(projection.matrix_cells[local]),
                    int(offsets[fracture_index] + local),
                ))
    return WellFractureConnections(
        well_index=[row[0] for row in rows],
        segment_index=[row[1] for row in rows],
        fracture_index=[row[2] for row in rows],
        points=np.asarray([row[3] for row in rows]).reshape(-1, 3),
        matrix_cells=[row[4] for row in rows],
        fracture_cells=[row[5] for row in rows],
    )


__all__ = ["well_fracture_nncs_3d"]
