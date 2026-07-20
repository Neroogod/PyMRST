"""MRST ``plotNodeDataDeformed.m`` node-data patch geometry."""

import numpy as np

from pymrst.modules.vemmech.plotting import (
    VEMPlotHandle,
    _boundary_faces,
    deformed_grid,
    node_patch_topology,
)


def plot_node_data_deformed_geometry(g, node_data, u):
    """Return patch geometry for deformed node data."""
    gd = deformed_grid(g, u)
    entities = _boundary_faces(gd) if gd.griddim == 3 else np.arange(gd.cells.num)
    faces, verts = node_patch_topology(gd, entities)
    vertices = gd.nodes.coords[verts, :]
    if faces.shape[0] == vertices.shape[0]:
        vertices = np.vstack([vertices, np.full((1, vertices.shape[1]), np.nan)])
    cdata = np.asarray(node_data)[verts, ...]
    return {
        "faces": faces,
        "vertices": vertices,
        "verts": verts,
        "cdata": cdata,
        "face_color": "interp",
    }


def plot_node_data_deformed(
    g, node_data, u, backend: str = "geometry", **backend_kwargs
):
    """Return a handle for MRST ``plotNodeDataDeformed``."""
    return VEMPlotHandle(
        "node_data_deformed",
        plot_node_data_deformed_geometry,
        g,
        u=np.asarray(u, dtype=float),
        data=np.asarray(node_data),
        backend=backend,
        backend_kwargs=backend_kwargs,
    )

__all__ = ["plot_node_data_deformed", "plot_node_data_deformed_geometry"]
