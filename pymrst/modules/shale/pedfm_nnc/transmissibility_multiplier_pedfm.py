"""Projection-based face transmissibility multiplier (`transmultpEDFM`)."""

import numpy as np

from pymrst.gridprocessing.grid import Grid


def transmissibility_multiplier_pedfm(
    grid: Grid,
    projected_matrix_neighbors,
    projected_areas,
    *,
    tolerance: float = 1e-10,
) -> np.ndarray:
    """Reduce matrix-face area by all projected fracture areas on that face.

    Cell pairs are orientation-insensitive and duplicate projection records are
    accumulated. Boundary faces cannot be projection matrix/matrix neighbors.
    """
    pairs = np.asarray(projected_matrix_neighbors, dtype=np.int64).reshape(-1, 2)
    areas = np.asarray(projected_areas, dtype=float).reshape(-1)
    if pairs.shape[0] != areas.size:
        raise ValueError("one projected area is required for each neighbor pair")
    if np.any(areas < 0) or not np.all(np.isfinite(areas)):
        raise ValueError("projected areas must be finite and non-negative")
    old = np.asarray(grid.faces.areas, dtype=float)
    new = old.copy()
    internal_faces = np.flatnonzero(np.all(grid.faces.neighbors >= 0, axis=1))
    lookup: dict[tuple[int, int], int] = {}
    for face in internal_faces:
        pair = tuple(sorted(map(int, grid.faces.neighbors[face])))
        if pair in lookup:
            raise ValueError(f"duplicate geometric neighbor pair {pair}")
        lookup[pair] = int(face)
    for pair, area in zip(pairs, areas):
        key = tuple(sorted(map(int, pair)))
        try:
            face = lookup[key]
        except KeyError as exc:
            raise ValueError(f"projected pair {tuple(pair)} is not a matrix face") from exc
        new[face] -= area
    new[new < tolerance] = 0.0
    if np.any(new < 0):
        raise ValueError("projected area exceeds geometric face area")
    return np.divide(new, old, out=np.ones_like(new), where=old > 0)


__all__ = ["transmissibility_multiplier_pedfm"]
