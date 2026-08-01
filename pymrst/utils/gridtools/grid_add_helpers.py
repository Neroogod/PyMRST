"""Attach MRST ``gridAddHelpers`` plotting and query closures to a grid."""

from __future__ import annotations

from types import SimpleNamespace
import numpy as np


def grid_add_helpers(grid):
    if getattr(grid.cells, "centroids", None) is None:
        raise ValueError("gridAddHelpers requires geometry; call compute_geometry first")

    def plot_grid(*args, **kwargs):
        from pymrst.plotting import plot_grid
        return plot_grid(grid, *args, **kwargs)

    def plot_cell_data(data, *args, **kwargs):
        from pymrst.plotting import plot_cell_data
        return plot_cell_data(grid, data, *args, **kwargs)

    def plot_faces(*args, **kwargs):
        from pymrst.plotting import plot_faces
        return plot_faces(grid, *args, **kwargs)

    def plot_points(points, ax=None, **kwargs):
        import matplotlib.pyplot as plt
        points = np.asarray(points, dtype=float)
        if ax is None:
            ax = plt.gca()
        style = {"color": "black", "marker": "."} | kwargs
        if points.shape[1] == 3 and hasattr(ax, "scatter3D"):
            return ax.scatter3D(points[:, 0], points[:, 1], points[:, 2], **style)
        return ax.scatter(points[:, 0], points[:, 1], **style)

    from pymrst.utils.gridtools.grid_cell_faces import grid_cell_faces
    from pymrst.utils.gridtools.grid_cell_no import grid_cell_no
    from pymrst.utils.gridtools.grid_cell_nodes import grid_cell_nodes
    from pymrst.utils.gridtools.grid_face_nodes import grid_face_nodes
    from pymrst.utils.gridtools.grid_logical_indices import grid_logical_indices

    grid.plot = SimpleNamespace(
        grid=plot_grid, cellData=plot_cell_data, faces=plot_faces,
        points=plot_points,
    )
    grid.helpers = SimpleNamespace(
        cellNo=lambda *a, **k: grid_cell_no(grid, *a, **k),
        getLogicalIndices=lambda *a, **k: grid_logical_indices(grid, *a, **k),
        getFaceNodes=lambda faces: grid_face_nodes(grid, faces),
        getCellNodes=lambda cells: grid_cell_nodes(grid, cells),
        getCellFaces=lambda cells: grid_cell_faces(grid, cells),
    )
    if "gridAddHelpers" not in grid.type:
        grid.type.append("gridAddHelpers")
    return grid


gridAddHelpers = grid_add_helpers

__all__ = ["gridAddHelpers", "grid_add_helpers"]
