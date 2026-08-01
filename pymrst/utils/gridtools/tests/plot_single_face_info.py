"""Inspect one grid face (MRST ``plotSingleFaceInfo.m``)."""

from __future__ import annotations
import numpy as np


def plot_single_face_info(grid, face: int, *, r_rel=.1, from_cell=None,
                          expand_fac=1.0, plot_info=True, ax=None):
    face = int(face)
    if not 0 <= face < grid.faces.num:
        raise IndexError("face index outside grid")
    centroid = np.asarray(grid.faces.centroids[face], float)
    start, stop = grid.faces.nodePos[face:face + 2]
    node_ids = np.asarray(grid.faces.nodes[start:stop], int)
    coordinates = np.asarray(grid.nodes.coords[node_ids], float)
    neighbors = np.asarray(grid.faces.neighbors[face], int)
    if from_cell is not None:
        from_cell = int(from_cell)
        if from_cell not in neighbors:
            raise ValueError(f"cell {from_cell} is not a neighbor of face {face}")
        shift = (float(expand_fac) - 1) * (centroid - grid.cells.centroids[from_cell])
        centroid, coordinates = centroid + shift, coordinates + shift
    normal = np.asarray(grid.faces.normals[face], float)
    unit_normal = normal / np.linalg.norm(normal) if np.linalg.norm(normal) else np.zeros_like(normal)
    payload = {
        "face": face, "centroid": centroid, "nodes": node_ids,
        "coordinates": coordinates, "neighbors": neighbors,
        "unit_normal": unit_normal,
        "zero_length_segments": int(node_ids.size - np.unique(coordinates, axis=0).shape[0]),
        "from_cell": from_cell,
    }
    if ax is not None:
        if coordinates.shape[1] == 3 and hasattr(ax, "scatter3D"):
            ax.scatter3D(coordinates[:, 0], coordinates[:, 1], coordinates[:, 2], color="black")
            ax.scatter3D(*centroid, color="black", marker="x")
            ax.quiver(*centroid, *unit_normal, length=max(float(r_rel), 1e-12), color="green")
        else:
            ax.scatter(coordinates[:, 0], coordinates[:, 1], color="black")
        if plot_info:
            ax.set_title(f"face {face}; neighbors {neighbors.tolist()}")
    return payload


plotSingleFaceInfo = plot_single_face_info
__all__ = ["plotSingleFaceInfo", "plot_single_face_info"]
