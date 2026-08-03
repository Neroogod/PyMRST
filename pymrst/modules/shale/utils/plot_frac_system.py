"""Plot an EDFM fracture system (MRST ``plotFracSystem.m``)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from pymrst.modules.shale.data import FracturePlane, ShaleGlobalGrid
from pymrst.plotting.plot_grid import plot_grid
from pymrst.plotting.private.rendering_compat import configure_reservoir_camera


@dataclass
class FractureSystemPlot:
    """MRST handle groups plus the shared PyVista plotter."""

    matrix: Any
    fractures: tuple[Any, ...]
    wells: tuple[Any, ...]
    intersections: tuple[Any, ...]
    completion_centroids: tuple[Any, ...]
    labels: tuple[Any, ...]
    plotter: Any

    def __iter__(self):
        # Enables the MRST-shaped ``hm, hf, hw = plot_frac_system(...)``.
        yield self.matrix
        yield self.fractures
        yield self.wells


def _field(value, name, default=None):
    if isinstance(value, Mapping):
        return value.get(name, default)
    return getattr(value, name, default)


def _points(value, field="points") -> np.ndarray:
    points = np.asarray(_field(value, field, np.empty((0, 3))), dtype=float)
    if points.size == 0:
        return np.empty((0, 3))
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"{field} must have shape (n, 3)")
    return points


def _fracture_centers(grid, ids: np.ndarray) -> np.ndarray:
    """Resolve MRST ``XFracCellIDs`` into displayed cell centroids."""
    if ids.size == 0:
        return np.empty((0, 3))
    if isinstance(grid, ShaleGlobalGrid):
        global_ids = ids[:, 0].astype(np.int64)
        local = global_ids - grid.n_matrix
        valid = (local >= 0) & (local < grid.n_fracture)
        return np.asarray(grid.fracture_centroids)[local[valid]]

    frac_grid = _field(grid, "FracGrid")
    if frac_grid is None:
        return np.empty((0, 3))
    centers = []
    for global_id, plane_id in ids.astype(np.int64):
        key = f"Frac{plane_id + 1}"
        frac = _field(frac_grid, key)
        if frac is None:
            continue
        cells = _field(frac, "cells", frac)
        start = int(_field(cells, "start", 0))
        centroids = np.asarray(_field(cells, "centroids"), dtype=float)
        local = int(global_id - start)
        if 0 <= local < len(centroids):
            centers.append(centroids[local])
    return np.asarray(centers, dtype=float).reshape(-1, 3)


def _scaled(points: np.ndarray, zscale: float) -> np.ndarray:
    out = np.array(points, dtype=float, copy=True)
    if out.size:
        out[:, 2] *= float(zscale)
    return out


def plot_frac_system(
    grid,
    fracture_planes: Sequence[FracturePlane | Mapping[str, Any]],
    wells=(),
    *,
    fracture_list=None,
    label: bool = True,
    fracture_face_alpha: float = 0.5,
    fracture_face_color="yellow",
    well_color="blue",
    plotter=None,
    filename: str | Path | None = None,
    show: bool = True,
    flip_z: bool = True,
    zscale: float = 1.0,
) -> FractureSystemPlot:
    """Render matrix grid, fracture polygons, wells and completion markers.

    ``fracture_list`` is zero-based, consistently with other PyMRST APIs.
    Wells may be objects or mappings with MRST fields ``points``,
    ``WellXFracPts`` and ``XFracCellIDs``.
    """
    try:
        import pyvista as pv
    except ImportError as error:  # pragma: no cover - optional dependency
        raise ImportError("plot_frac_system requires PyVista; install pymrst[plotting]") from error

    planes = tuple(fracture_planes)
    selected = (np.arange(len(planes), dtype=np.int64) if fracture_list is None
                else np.asarray(fracture_list, dtype=np.int64).reshape(-1))
    if np.any((selected < 0) | (selected >= len(planes))):
        raise IndexError("fracture_list contains an out-of-range fracture index")
    own_plotter = plotter is None
    target = plotter or pv.Plotter(off_screen=filename is not None)

    fracture_actors, label_actors = [], []
    for index in selected:
        points = _scaled(_points(planes[int(index)]), zscale)
        faces = np.r_[len(points), np.arange(len(points), dtype=np.int64)]
        mesh = pv.PolyData(points, faces=faces)
        color = fracture_face_color
        if (isinstance(fracture_face_color, (list, tuple, np.ndarray))
                and len(fracture_face_color) == len(planes)
                and not (len(fracture_face_color) == 3
                         and all(np.isscalar(v) for v in fracture_face_color))):
            color = fracture_face_color[int(index)]
        fracture_actors.append(target.add_mesh(
            mesh, color=color, opacity=float(fracture_face_alpha),
            show_edges=True,
        ))
        if label:
            anchor = points[np.argmax(points[:, 0])]
            label_actors.append(target.add_point_labels(
                anchor[None, :], [f"Fracture {int(index) + 1}"],
                text_color="red", point_size=0, shape=None,
            ))

    matrix_grid = grid.matrix if isinstance(grid, ShaleGlobalGrid) else grid
    matrix = plot_grid(
        matrix_grid, face_color="none", edge_color="black", opacity=0.5,
        plotter=target, show=False, flip_z=flip_z, zscale=zscale,
    )

    well_actors, intersection_actors, centroid_actors = [], [], []
    if wells is None:
        wells = ()
    if isinstance(wells, Mapping) or hasattr(wells, "points"):
        wells = (wells,)
    for well in wells:
        trajectory = _scaled(_points(well), zscale)
        if len(trajectory) >= 2:
            well_actors.append(target.add_mesh(
                pv.lines_from_points(trajectory), color=well_color, line_width=2,
            ))
        intersections = _scaled(_points(well, "WellXFracPts"), zscale)
        if len(intersections):
            intersection_actors.append(target.add_mesh(
                pv.PolyData(intersections), color=(0, .75, .75),
                point_size=10, render_points_as_spheres=True,
            ))
        ids = np.asarray(_field(well, "XFracCellIDs", np.empty((0, 2))))
        if ids.size:
            ids = ids.reshape(-1, 2)
            centers = _scaled(_fracture_centers(grid, ids), zscale)
            if len(centers):
                centroid_actors.append(target.add_mesh(
                    pv.PolyData(centers), color=(0, .447, .741),
                    point_size=10, render_points_as_spheres=False,
                ))

    target.camera_position = "iso"
    configure_reservoir_camera(target, flip_z=flip_z, reset=own_plotter)
    if filename is not None:
        target.screenshot(str(filename))
        if own_plotter:
            target.close()
    elif show and own_plotter:
        target.show()
    return FractureSystemPlot(
        matrix, tuple(fracture_actors), tuple(well_actors),
        tuple(intersection_actors), tuple(centroid_actors), tuple(label_actors), target,
    )


plotFracSystem = plot_frac_system

__all__ = ["FractureSystemPlot", "plot_frac_system", "plotFracSystem"]
