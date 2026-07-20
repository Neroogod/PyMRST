"""MRST ``plotFaces2D.m`` 2D face-line geometry."""

from typing import Optional

import numpy as np

from pymrst.modules.vemmech.plotting import VEMPlotHandle, _array_or_none, deformed_grid


def plot_faces_2d_geometry(
    g,
    faces: np.ndarray,
    uu: Optional[np.ndarray] = None,
    shift: Optional[np.ndarray] = None,
):
    """Return line geometry for 2D faces."""
    if g.griddim != 2:
        raise ValueError("plot_faces_2d_geometry only supports 2D grids")
    gd = deformed_grid(g, uu) if uu is not None else g
    faces = np.asarray(faces, dtype=np.int64).ravel()
    shift_arr = None if shift is None else np.asarray(shift, dtype=float).ravel()
    lines = []
    for j, face in enumerate(faces):
        nodes = gd.faces.nodes[gd.faces.nodePos[face]:gd.faces.nodePos[face + 1]]
        pos = gd.nodes.coords[nodes, :2].copy()
        if shift_arr is not None:
            pos += gd.faces.normals[face, :2] * shift_arr[j]
        lines.append(pos)
    return np.asarray(lines)


def plot_faces_2d(
    g,
    faces: np.ndarray,
    uu: Optional[np.ndarray] = None,
    shift: Optional[np.ndarray] = None,
    backend: str = "geometry",
    **backend_kwargs,
):
    """Return a handle for MRST ``plotFaces2D``."""
    if backend != "geometry":
        raise ValueError("plot_faces_2d currently supports backend='geometry'")
    return VEMPlotHandle(
        "faces_2d",
        plot_faces_2d_geometry,
        g,
        u=_array_or_none(uu),
        entities=np.asarray(faces, dtype=np.int64),
        shift=_array_or_none(shift),
        backend=backend,
        backend_kwargs=backend_kwargs,
    )

__all__ = ["plot_faces_2d", "plot_faces_2d_geometry"]
