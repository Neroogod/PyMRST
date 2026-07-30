"""Symmetric reverse Cuthill–McKee ordering for grid cells."""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp


def _get(obj, name):
    return obj[name] if isinstance(obj, dict) else getattr(obj, name)


def get_grid_symrcm_ordering(grid):
    """Return zero-based cell ordering from internal face connectivity."""
    neighbors = np.asarray(
        _get(_get(grid, "faces"), "neighbors"), dtype=np.int64
    )
    native = np.any(neighbors < 0)
    if native:
        internal = np.all(neighbors >= 0, axis=1)
        pairs = neighbors[internal]
        number_of_cells = int(_get(_get(grid, "cells"), "num"))
    else:
        internal = np.all(neighbors > 0, axis=1)
        pairs = neighbors[internal] - 1
        number_of_cells = int(_get(_get(grid, "cells"), "num"))
    rows = np.concatenate([pairs[:, 0], pairs[:, 1]]) if pairs.size else np.array([], int)
    columns = np.concatenate([pairs[:, 1], pairs[:, 0]]) if pairs.size else np.array([], int)
    adjacency = sp.csr_matrix(
        (np.ones(rows.size), (rows, columns)),
        shape=(number_of_cells, number_of_cells),
    )
    adjacency.setdiag(0)
    adjacency.eliminate_zeros()
    neighbors_by_cell = [
        adjacency.indices[adjacency.indptr[cell]:adjacency.indptr[cell + 1]]
        for cell in range(number_of_cells)
    ]
    degree = np.asarray(adjacency.getnnz(axis=1)).ravel()

    def level_structure(root, allowed):
        levels = []
        seen = {int(root)}
        frontier = [int(root)]
        while frontier:
            levels.append(frontier)
            following = []
            for cell in frontier:
                for neighbor in neighbors_by_cell[cell]:
                    neighbor = int(neighbor)
                    if neighbor in allowed and neighbor not in seen:
                        seen.add(neighbor)
                        following.append(neighbor)
            frontier = following
        return levels

    remaining = set(range(number_of_cells))
    result = []
    while remaining:
        root = min(remaining, key=lambda cell: (degree[cell], cell))
        levels = level_structure(root, remaining)
        while levels:
            candidate = min(
                levels[-1], key=lambda cell: (degree[cell], cell)
            )
            candidate_levels = level_structure(candidate, remaining)
            if len(candidate_levels) <= len(levels):
                root = candidate
                break
            root, levels = candidate, candidate_levels
        queue = [root]
        seen = {root}
        cuthill = []
        while queue:
            cell = queue.pop(0)
            cuthill.append(cell)
            unvisited = [
                int(neighbor)
                for neighbor in neighbors_by_cell[cell]
                if int(neighbor) in remaining and int(neighbor) not in seen
            ]
            unvisited.sort(key=lambda neighbor: (degree[neighbor], neighbor))
            seen.update(unvisited)
            queue.extend(unvisited)
        result.extend(reversed(cuthill))
        remaining.difference_update(seen)
    return np.asarray(result, dtype=np.int64)


getGridSYMRCMOrdering = get_grid_symrcm_ordering

__all__ = ["get_grid_symrcm_ordering", "getGridSYMRCMOrdering"]
