"""Structured multidimensional interpolation with AD chain rules."""

from __future__ import annotations

import numpy as np
from scipy.interpolate import RegularGridInterpolator

from pymrst.modules.ad_core.utils.double_to_adi import double_to_adi
from pymrst.modules.ad_core.utils.get_sample_ad import get_sample_ad
from pymrst.utils.adi import value


def get_multi_dim_interpolator(points, table, extrapolation="linear"):
    """Return ``(function, interpolator, epsilons)`` as in MRST.

    Grid vectors may be ascending or descending. AD arguments receive
    numerical slopes evaluated with MRST's forward half-spacing difference.
    """
    axes = [np.asarray(axis, dtype=float).ravel() for axis in points]
    values = np.asarray(table, dtype=float)
    if len(axes) > 1 and values.shape != tuple(axis.size for axis in axes):
        raise ValueError("table dimensions must match interpolation axes")
    for dimension, axis in enumerate(axes):
        if axis.size < 2:
            raise ValueError("each interpolation axis needs at least two points")
        if axis[1] < axis[0]:
            axes[dimension] = axis[::-1]
            values = np.flip(values, axis=dimension)
    epsilons = np.asarray(
        [np.min(np.abs(np.diff(axis))) / 2.0 for axis in axes], dtype=float
    )
    if np.min(epsilons) <= 1e-18:
        raise ValueError("difference between points must be larger than 2e-18")
    fill_value = np.nan if extrapolation in ("none", None) else None
    interpolator = RegularGridInterpolator(
        tuple(axes), values, method="linear", bounds_error=False, fill_value=fill_value
    )

    def evaluate(*arguments):
        if len(arguments) != len(axes):
            raise ValueError("one query argument is required per dimension")
        numeric = [np.asarray(value(argument), dtype=float) for argument in arguments]
        broadcast = np.broadcast_arrays(*numeric)
        query = np.column_stack([item.ravel() for item in broadcast])
        base = interpolator(query).reshape(broadcast[0].shape)
        sample, is_ad = get_sample_ad(*arguments)
        if not is_ad:
            return base
        result = double_to_adi(base.ravel(), sample)
        for dimension, argument in enumerate(arguments):
            if not (hasattr(argument, "val") and hasattr(argument, "jac")):
                continue
            shifted = [item.copy() for item in broadcast]
            shifted[dimension] += epsilons[dimension]
            shifted_query = np.column_stack([item.ravel() for item in shifted])
            slope = (
                interpolator(shifted_query).reshape(base.shape) - base
            ) / epsilons[dimension]
            centered = argument - np.asarray(value(argument), dtype=float)
            result = result + centered * slope.ravel()
        return result

    return evaluate, interpolator, epsilons


getMultiDimInterpolator = get_multi_dim_interpolator

__all__ = ["get_multi_dim_interpolator", "getMultiDimInterpolator"]
