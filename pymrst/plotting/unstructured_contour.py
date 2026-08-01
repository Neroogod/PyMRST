"""Contour scalar data over a 2-D unstructured grid."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from pymrst.plotting.interpolate_data import interpolate_data
from pymrst.utils.compute_well_index import cell_dims


@dataclass
class ContourPlotResult:
    x: np.ndarray
    y: np.ndarray
    values: np.ndarray
    cart_dims: tuple[int, int]
    levels: Any
    filled: bool
    contour_set: Any = None
    axes: Any = None


def contour_grid(
    g,
    data: Any,
    *,
    cart_dims=None,
    extrapolation: str = "linear",
):
    """Sample unstructured scalar data on an MRST-style Cartesian grid."""
    if g.griddim != 2:
        raise ValueError("unstructured_contour supports only 2-D grids")
    values = np.asarray(data, dtype=float)
    if values.ndim == 2 and values.shape[1] == 1:
        values = values[:, 0]
    if values.ndim != 1:
        raise ValueError("unstructured_contour requires scalar data")
    coordinates = np.asarray(g.nodes.coords, dtype=float)
    lower = coordinates.min(axis=0)
    upper = coordinates.max(axis=0)
    if cart_dims is None:
        dimensions = cell_dims(g, np.arange(g.cells.num))[:, :2]
        spacing = np.min(dimensions, axis=0)
        inferred = np.ceil((upper-lower)/spacing).astype(np.int64)
        dims = tuple(np.clip(inferred, 2, 1000).tolist())
    else:
        raw = np.asarray(cart_dims, dtype=float).ravel()
        if raw.size != 2 or not np.all(np.isfinite(raw)):
            raise ValueError("cart_dims must contain two finite integers")
        if not np.all(raw == np.floor(raw)) or np.any(raw < 2):
            raise ValueError("cart_dims entries must be integers of at least 2")
        dims = tuple(np.minimum(raw.astype(np.int64), 1000).tolist())
    x, y = np.meshgrid(
        np.linspace(lower[0], upper[0], dims[0]),
        np.linspace(lower[1], upper[1], dims[1]),
        indexing="ij",
    )
    query = np.column_stack([x.ravel(order="F"), y.ravel(order="F")])
    sampled = interpolate_data(
        g,
        values,
        xq=query,
        interpolation="linear",
        extrapolation=extrapolation,
    ).reshape(dims, order="F")
    return x, y, sampled, dims


def unstructured_contour(
    g,
    data: Any,
    levels=None,
    *,
    cart_dims=None,
    fill: bool = False,
    extrapolation: str = "linear",
    axes=None,
    **contour_kwargs,
) -> ContourPlotResult:
    """Draw filled/unfilled contours after structured resampling."""
    x, y, sampled, dims = contour_grid(
        g, data, cart_dims=cart_dims, extrapolation=extrapolation
    )
    requested = 10 if levels is None else levels
    try:
        import matplotlib.pyplot as plt
    except ImportError as error:  # pragma: no cover
        raise ImportError(
            "unstructured_contour requires Matplotlib; "
            "install pymrst[plotting]"
        ) from error
    target = axes
    if target is None:
        _, target = plt.subplots()
    renderer = target.contourf if fill else target.contour
    contour_set = renderer(x, y, sampled, requested, **contour_kwargs)
    return ContourPlotResult(
        x=x,
        y=y,
        values=sampled,
        cart_dims=dims,
        levels=np.asarray(contour_set.levels).copy(),
        filled=bool(fill),
        contour_set=contour_set,
        axes=target,
    )


unstructuredContour = unstructured_contour

__all__ = [
    "ContourPlotResult",
    "contour_grid",
    "unstructured_contour",
    "unstructuredContour",
]
