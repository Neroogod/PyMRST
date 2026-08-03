"""Clean and classify stochastic fracture polygons."""

from collections.abc import Sequence

import numpy as np


def _polygon_area_3d(points: np.ndarray) -> float:
    shifted = np.roll(points, -1, axis=0)
    return 0.5 * float(np.linalg.norm(np.sum(np.cross(points, shifted), axis=0)))


def process_stochastic_fractures(
    fracture_set: Sequence[np.ndarray],
    *,
    area_tolerance: float = 1e-5,
    planar_tolerance: float = 1e-9,
) -> tuple[list[np.ndarray], list[np.ndarray], np.ndarray]:
    """Port the filtering contract of MRST ``processStochFracs``.

    MRST input polygons carry a final terminator/closing row, which is excluded
    from area and planarity calculations but retained in returned polygons.
    """
    planar: list[np.ndarray] = []
    nonplanar: list[np.ndarray] = []
    areas = np.zeros(len(fracture_set), dtype=float)
    for index, raw in enumerate(fracture_set):
        polygon = np.asarray(raw, dtype=float)
        if polygon.ndim != 2 or polygon.shape[1] != 3 or polygon.shape[0] < 2:
            continue
        vertices = polygon[:-1]
        if np.isnan(vertices).any() or vertices.shape[0] < 3:
            continue
        areas[index] = _polygon_area_3d(vertices)
        if areas[index] < area_tolerance:
            continue
        centered = vertices - vertices.mean(axis=0)
        singular = np.linalg.svd(centered, compute_uv=False)
        scale = max(float(singular[0]), 1.0)
        if singular[-1] <= planar_tolerance * scale:
            planar.append(polygon.copy())
        else:
            nonplanar.append(polygon.copy())
    return planar, nonplanar, areas


__all__ = ["process_stochastic_fractures"]
