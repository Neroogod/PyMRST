"""Connected regions separated by slice faces (MRST ``markCutGrids``)."""

from __future__ import annotations

from collections import deque

import numpy as np
import scipy.sparse as sp
from scipy.sparse.csgraph import connected_components


def _adjacency(grid, face_status, faces_along_plane: int):
    status = np.asarray(face_status).ravel()
    if status.shape != (grid.faces.num,):
        raise ValueError("face_status must contain one value per grid face")
    neighbors = np.asarray(grid.faces.neighbors, dtype=np.int64)
    keep = status != faces_along_plane
    pairs = neighbors[keep]
    pairs = pairs[np.all(pairs >= 0, axis=1)]
    nc = grid.cells.num
    if pairs.size:
        rows = np.concatenate([pairs[:, 0], pairs[:, 1]])
        cols = np.concatenate([pairs[:, 1], pairs[:, 0]])
        graph = sp.csr_array(
            (np.ones(rows.size), (rows, cols)), shape=(nc, nc)
        )
    else:
        graph = sp.csr_array((nc, nc))
    return graph


def mark_cut_grids(
    grid,
    face_status,
    *,
    faces_along_plane: int = 3,
    legacy: bool = False,
    start=None,
) -> np.ndarray:
    """Label cells connected without crossing the selected cut faces.

    The default returns zero-based connected-component labels. In legacy mode
    it returns a 0/1 mask for the component containing ``start``. ``start`` may
    be a zero-based cell index or a coordinate; the nearest cell centroid is
    used for coordinates, matching the normal use of MRST's enclosing-cell
    lookup on valid cell centroids.
    """
    graph = _adjacency(grid, face_status, faces_along_plane)
    if not legacy:
        _count, labels = connected_components(
            graph, directed=False, return_labels=True
        )
        return labels.astype(np.int64)

    if start is None:
        start_cell = 0
    else:
        value = np.asarray(start)
        if value.ndim == 0:
            start_cell = int(value)
        else:
            if grid.cells.centroids is None:
                raise ValueError("coordinate start requires cell centroids")
            point = np.asarray(start, dtype=float).ravel()
            centroids = np.asarray(grid.cells.centroids, dtype=float)
            if point.size != centroids.shape[1]:
                raise ValueError("start coordinate has the wrong dimension")
            start_cell = int(np.argmin(np.linalg.norm(
                centroids - point[None, :], axis=1
            )))
    if start_cell < 0 or start_cell >= grid.cells.num:
        raise IndexError("start cell is out of range")

    mask = np.zeros(grid.cells.num, dtype=np.int64)
    indptr, indices = graph.indptr, graph.indices
    queue = deque([start_cell])
    mask[start_cell] = 1
    while queue:
        cell = queue.popleft()
        for other in indices[indptr[cell]:indptr[cell + 1]]:
            if not mask[other]:
                mask[other] = 1
                queue.append(int(other))
    return mask


markCutGrids = mark_cut_grids

__all__ = ["mark_cut_grids", "markCutGrids"]
