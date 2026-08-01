"""Interpolate grid data (MRST plotting/interpolateData.m)."""

from __future__ import annotations

from typing import Any

import numpy as np


_LOCATIONS = {"nodes", "faces", "cells"}


def _coordinates(g, location: str) -> np.ndarray:
    key = location.lower()
    if key == "nodes":
        coordinates = g.nodes.coords
    elif key == "faces":
        coordinates = g.faces.centroids
    elif key == "cells":
        coordinates = g.cells.centroids
    else:
        raise ValueError("query must be 'nodes', 'faces', or 'cells'")
    if coordinates is None:
        raise ValueError(f"grid geometry for {key} has not been computed")
    result = np.asarray(coordinates, dtype=float)
    if result.ndim != 2 or result.shape[1] != g.griddim:
        raise ValueError(f"{key} coordinates must have G.griddim columns")
    return result


def _source_coordinates(g, rows: int, explicit) -> np.ndarray:
    if explicit is not None and np.asarray(explicit).size:
        coordinates = np.asarray(explicit, dtype=float)
    else:
        matches = [
            name for name, count in (
                ("nodes", g.nodes.num),
                ("faces", g.faces.num),
                ("cells", g.cells.num),
            ) if count == rows
        ]
        if not matches:
            raise ValueError(
                "data coordinates were not provided and row count does not "
                "match grid nodes, faces, or cells"
            )
        if len(matches) > 1:
            raise ValueError(
                "data row count is ambiguous between " + ", ".join(matches)
                + "; provide explicit source coordinates"
            )
        coordinates = _coordinates(g, matches[0])
    if coordinates.ndim != 2 or coordinates.shape != (rows, g.griddim):
        raise ValueError(
            f"source coordinates must have shape ({rows}, {g.griddim})"
        )
    if not np.all(np.isfinite(coordinates)):
        raise ValueError("source coordinates must be finite")
    return coordinates


def _query_coordinates(g, query) -> np.ndarray:
    if query is None or (not isinstance(query, str) and np.asarray(query).size == 0):
        return _coordinates(g, "nodes")
    if isinstance(query, str):
        return _coordinates(g, query)
    coordinates = np.asarray(query, dtype=float)
    if coordinates.ndim != 2 or coordinates.shape[1] != g.griddim:
        raise ValueError(
            f"query coordinates must have {g.griddim} columns"
        )
    return coordinates


def _linear_extrapolation(points, values, query, triangulation, outside):
    output = np.empty(outside.size, dtype=float)
    simplices = triangulation.simplices
    centers = points[simplices].mean(axis=1)
    for ordinal, query_index in enumerate(outside):
        point = query[query_index]
        simplex = simplices[
            np.argmin(np.sum((centers-point)**2, axis=1))
        ]
        design = np.column_stack([points[simplex], np.ones(simplex.size)])
        coefficients = np.linalg.lstsq(
            design, values[simplex], rcond=None
        )[0]
        output[ordinal] = np.r_[point, 1.0] @ coefficients
    return output


def _interpolate_column(points, values, query, interpolation, extrapolation):
    from scipy.interpolate import (
        CloughTocher2DInterpolator,
        LinearNDInterpolator,
        NearestNDInterpolator,
    )
    from scipy.spatial import Delaunay

    method = interpolation.lower()
    extrap = extrapolation.lower()
    if method not in {"linear", "nearest", "natural"}:
        raise ValueError("interpolation must be linear, nearest, or natural")
    if extrap not in {"linear", "nearest", "none", "nan"}:
        raise ValueError("extrapolation must be linear, nearest, or none")
    if points.shape[1] == 1:
        order = np.argsort(points[:, 0])
        x, y = points[order, 0], values[order]
        if method == "nearest":
            indices = np.abs(query[:, 0, None]-x[None, :]).argmin(axis=1)
            return y[indices]
        result = np.interp(query[:, 0], x, y, left=np.nan, right=np.nan)
        outside = np.flatnonzero(np.isnan(result))
        if extrap == "nearest":
            result[query[:, 0] < x[0]] = y[0]
            result[query[:, 0] > x[-1]] = y[-1]
        elif extrap == "linear" and x.size >= 2:
            left = query[:, 0] < x[0]
            right = query[:, 0] > x[-1]
            result[left] = y[0]+(query[left, 0]-x[0])*(y[1]-y[0])/(x[1]-x[0])
            result[right] = y[-1]+(query[right, 0]-x[-1])*(y[-1]-y[-2])/(x[-1]-x[-2])
        return result

    triangulation = Delaunay(points)
    if method == "nearest":
        return np.asarray(NearestNDInterpolator(points, values)(query))
    if method == "natural":
        if points.shape[1] != 2:
            raise ValueError("natural interpolation is supported only in 2-D")
        result = np.asarray(
            CloughTocher2DInterpolator(points, values, fill_value=np.nan)(query)
        )
    else:
        result = np.asarray(
            LinearNDInterpolator(triangulation, values, fill_value=np.nan)(query)
        )
    outside = np.flatnonzero(np.isnan(result))
    if outside.size and extrap == "nearest":
        result[outside] = NearestNDInterpolator(points, values)(query[outside])
    elif outside.size and extrap == "linear":
        result[outside] = _linear_extrapolation(
            points, values, query, triangulation, outside
        )
    return result


def interpolate_data(
    g,
    data: Any,
    x=None,
    xq="nodes",
    *,
    interpolation: str = "linear",
    extrapolation: str = "linear",
) -> np.ndarray:
    """Interpolate one or more data columns onto grid/query coordinates."""
    values = np.asarray(data, dtype=float)
    was_vector = values.ndim == 1
    if was_vector:
        values = values[:, None]
    if values.ndim != 2 or values.shape[0] == 0:
        raise ValueError("data must be a nonempty vector or matrix")
    source = _source_coordinates(g, values.shape[0], x)
    query = _query_coordinates(g, xq)
    output = np.column_stack([
        _interpolate_column(
            source,
            values[:, column],
            query,
            interpolation,
            extrapolation,
        )
        for column in range(values.shape[1])
    ])
    return output[:, 0] if was_vector else output


interpolateData = interpolate_data

__all__ = ["interpolate_data", "interpolateData"]
