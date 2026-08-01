"""Expand coarse faces to their fine-grid contributors."""

from __future__ import annotations

import numpy as np


def coarse_grid_parts(grid):
    """Return ``(coarse, parent, partition)`` or ``None`` for a fine grid."""
    if hasattr(grid, "g") and hasattr(grid, "partition"):
        coarse = grid.g
        parent = getattr(grid, "parent", None)
        if parent is None:
            parent = getattr(coarse, "parent", None)
        if parent is None:
            raise ValueError("coarse grid does not retain its fine parent")
        return coarse, parent, np.asarray(grid.partition, dtype=np.int64)
    parent = getattr(grid, "parent", None)
    partition = getattr(grid, "partition", None)
    if parent is not None and partition is not None:
        return grid, parent, np.asarray(partition, dtype=np.int64)
    return None


def _mapped_neighbors(parent, partition) -> np.ndarray:
    neighbors = np.asarray(parent.faces.neighbors, dtype=np.int64)
    mapped = np.full_like(neighbors, -1)
    active = neighbors >= 0
    mapped[active] = partition[neighbors[active]]
    return mapped


def expand_coarse_faces(grid, faces):
    """Return fine parent, fine faces, and selected-coarse-face group IDs."""
    parts = coarse_grid_parts(grid)
    if parts is None:
        selected = np.asarray(faces, dtype=np.int64).ravel()
        return grid, selected, np.arange(selected.size, dtype=np.int64)
    coarse, parent, partition = parts
    selected = np.asarray(faces, dtype=np.int64).ravel()
    mapped = _mapped_neighbors(parent, partition)
    fine_centroids = np.asarray(parent.faces.centroids, dtype=float)
    fine_normals = np.asarray(parent.faces.normals, dtype=float)
    coarse_centroids = np.asarray(coarse.faces.centroids, dtype=float)
    coarse_normals = np.asarray(coarse.faces.normals, dtype=float)
    extent = np.ptp(np.asarray(parent.nodes.coords, dtype=float), axis=0)
    tolerance = max(float(np.linalg.norm(extent)), 1.0)*1e-10

    expanded: list[np.ndarray] = []
    groups: list[np.ndarray] = []
    for ordinal, face in enumerate(selected):
        target_pair = np.sort(
            np.asarray(coarse.faces.neighbors[face], dtype=np.int64)
        )
        candidates = np.flatnonzero(
            np.all(np.sort(mapped, axis=1) == target_pair, axis=1)
        )
        normal = coarse_normals[face]
        n_norm = np.linalg.norm(normal)
        if n_norm == 0.0:
            raise ValueError(f"coarse face {face} has a zero normal")
        unit = normal/n_norm
        distance = np.abs(
            (fine_centroids[candidates]-coarse_centroids[face]) @ unit
        )
        fn = fine_normals[candidates]
        fn_norm = np.linalg.norm(fn, axis=1)
        parallel = np.abs(fn @ unit) >= (1.0-1e-8)*fn_norm
        contributors = candidates[(distance <= tolerance) & parallel]
        if contributors.size == 0:
            raise ValueError(
                f"could not map coarse face {face} to fine faces"
            )
        expanded.append(contributors)
        groups.append(np.full(contributors.size, ordinal, dtype=np.int64))
    if not expanded:
        return (
            parent,
            np.empty(0, dtype=np.int64),
            np.empty(0, dtype=np.int64),
        )
    return parent, np.concatenate(expanded), np.concatenate(groups)


__all__ = ["coarse_grid_parts", "expand_coarse_faces"]
