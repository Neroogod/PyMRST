"""MATLAB property and reservoir-view compatibility for VTK renderers."""

from __future__ import annotations

from dataclasses import replace

import numpy as np


_MATLAB_PROPERTIES = {
    "facecolor": "color",
    "edgecolor": "edge_color",
    "facealpha": "opacity",
    "linewidth": "line_width",
}


def translate_matlab_properties(properties):
    """Translate common MATLAB patch names to PyVista keyword names.

    Matching is case-insensitive. Native PyVista names take precedence when
    both forms are supplied.
    """
    translated = {}
    native = set(properties)
    for key, value in properties.items():
        mapped = _MATLAB_PROPERTIES.get(str(key).replace("_", "").lower())
        if mapped is None:
            translated[key] = value
        elif key == mapped:
            translated[key] = value
        elif mapped not in native:
            translated[mapped] = value
    edge = translated.get("edge_color")
    if isinstance(edge, str) and edge.lower() == "none":
        translated["edge_color"] = None
        translated.setdefault("show_edges", False)
    return translated


def scaled_patch_geometry(geometry, grid, zscale=1.0):
    """Scale depth about the grid's shallowest node, preserving topology."""
    scale = float(zscale)
    if not np.isfinite(scale) or scale <= 0.0:
        raise ValueError("zscale must be finite and positive")
    if geometry.vertices.size == 0 or geometry.vertices.shape[1] < 3 or np.isclose(scale, 1.0):
        return geometry
    vertices = np.asarray(geometry.vertices, dtype=float).copy()
    z0 = float(np.min(np.asarray(grid.nodes.coords, dtype=float)[:, 2]))
    vertices[:, 2] = z0 + scale*(vertices[:, 2] - z0)
    return replace(geometry, vertices=vertices)


def configure_reservoir_camera(plotter, *, flip_z=True, reset=False):
    """Apply the canonical MRST depth-down camera without changing geometry."""
    if not flip_z or plotter is None or not hasattr(plotter, "camera"):
        return plotter
    if reset and hasattr(plotter, "reset_camera"):
        plotter.reset_camera()
    if reset and hasattr(plotter, "camera_position"):
        plotter.camera_position = "xz"
    camera = plotter.camera
    if hasattr(camera, "up"):
        camera.up = (0.0, 0.0, -1.0)
    elif hasattr(camera, "view_up"):
        camera.view_up = (0.0, 0.0, -1.0)
    if reset and hasattr(camera, "azimuth"):
        camera.azimuth = 210
    if reset and hasattr(camera, "elevation"):
        camera.elevation = 25
    return plotter


__all__ = [
    "translate_matlab_properties",
    "scaled_patch_geometry",
    "configure_reservoir_camera",
]
