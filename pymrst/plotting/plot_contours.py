"""Discrete 3-D cell-data contours (MRST plotting/plotContours.m)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from pymrst.plotting.plot_faces import plot_faces
from pymrst.plotting.results import PlotResult


@dataclass
class ContourBand:
    lower: float
    upper: float
    color: Any
    exterior_faces: np.ndarray
    internal_faces: np.ndarray
    exterior_actor: PlotResult
    internal_actor: PlotResult


@dataclass
class GridContourResult:
    levels: np.ndarray
    bands: tuple[ContourBand, ...]
    plotter: Any = None


def contour_face_sets(g, values, count: int):
    """Return MRST exterior bands and internal threshold face sets."""
    if g.griddim != 3:
        raise ValueError("plot_contours supports only 3-D grids")
    data = np.asarray(values, dtype=float).ravel()
    if data.size != g.cells.num:
        raise ValueError(
            f"contour data has {data.size} values; expected {g.cells.num}"
        )
    if not np.all(np.isfinite(data)):
        raise ValueError("contour data must be finite")
    if isinstance(count, bool) or int(count) != count or int(count) <= 0:
        raise ValueError("count must be a positive integer")
    count = int(count)
    levels = np.linspace(float(data.min()), float(data.max()), count+1)
    neighbors = np.asarray(g.faces.neighbors, dtype=np.int64)
    exterior = np.any(neighbors < 0, axis=1)

    def touches(mask):
        result = np.zeros(g.faces.num, dtype=bool)
        for side in range(2):
            inside = neighbors[:, side] >= 0
            result[inside] |= mask[neighbors[inside, side]]
        return result

    sets = []
    for index in range(count):
        in_band = (data >= levels[index]) & (data <= levels[index+1])
        exterior_faces = np.flatnonzero(touches(in_band) & exterior)
        below = touches(data < levels[index+1])
        above = touches(data > levels[index+1])
        internal_faces = np.flatnonzero(below & above)
        sets.append((exterior_faces, internal_faces))
    return levels, tuple(sets)


def _colors(count, cmap):
    try:
        from matplotlib import colormaps
    except ImportError as error:  # pragma: no cover
        raise ImportError(
            "plot_contours colors require Matplotlib; "
            "install pymrst[plotting]"
        ) from error
    mapping = colormaps.get_cmap(cmap)
    positions = np.linspace(0.0, 1.0, count)
    return [mapping(position)[:3] for position in positions]


def plot_contours(
    g,
    values,
    count: int,
    *,
    cmap: str = "viridis",
    opacity: float = 0.4,
    plotter=None,
    show: bool = True,
    **mesh_kwargs,
) -> GridContourResult:
    """Plot exterior value bands and internal discrete level faces."""
    levels, face_sets = contour_face_sets(g, values, count)
    colors = _colors(int(count), cmap)
    target = plotter
    bands = []
    for index, (exterior, internal) in enumerate(face_sets):
        exterior_result = plot_faces(
            g,
            exterior,
            colors[index],
            edge_color=None,
            opacity=opacity,
            outline=True,
            plotter=target,
            show=False,
            **mesh_kwargs,
        )
        target = exterior_result.plotter
        internal_result = plot_faces(
            g,
            internal,
            colors[index],
            edge_color=None,
            opacity=opacity,
            outline=True,
            plotter=target,
            show=False,
            **mesh_kwargs,
        )
        target = internal_result.plotter
        bands.append(ContourBand(
            lower=float(levels[index]),
            upper=float(levels[index+1]),
            color=colors[index],
            exterior_faces=exterior,
            internal_faces=internal,
            exterior_actor=exterior_result,
            internal_actor=internal_result,
        ))
    if show and plotter is None and target is not None:
        target.show()
    return GridContourResult(levels, tuple(bands), target)


plotContours = plot_contours

__all__ = [
    "ContourBand",
    "GridContourResult",
    "contour_face_sets",
    "plot_contours",
    "plotContours",
]
