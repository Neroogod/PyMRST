"""MRST ``ad-core/utils/getMultiDimInterpolator.m``: a linear interpolant on
a structured grid of any dimension, for numbers or AD values.

Axes given in decreasing order are reversed with the table. Outside the grid
the interpolant extrapolates by `extrap` (MATLAB's `griddedInterpolant`:
`'linear'`, `'nearest'` or `'none'`); with an AD argument MRST switches to
linear extrapolation, and the derivative with respect to each AD argument is
the forward difference over half the smallest grid spacing (`epsilons`).
Returns `(fn, F, epsilons)`.
"""

from __future__ import annotations

from typing import Any, Callable, Sequence, Tuple

import numpy as np
from scipy.interpolate import RegularGridInterpolator


def get_multi_dim_interpolator(x: Sequence[Any], Fx: Any,
                               extrap: str = "linear") -> Tuple[Callable, Any, np.ndarray]:
    """MRST `[fn, F, epsilons] = getMultiDimInterpolator(x, Fx, extrap)`."""
    axes = [np.asarray(a, dtype=float).ravel() for a in x]
    Fx = np.asarray(Fx, dtype=float)
    if len(axes) > 1:
        assert Fx.ndim == len(axes)
        assert all(a.size == n for a, n in zip(axes, Fx.shape))
    for i, a in enumerate(axes):
        if a[1] - a[0] < 0:
            axes[i] = a[::-1]
            Fx = np.flip(Fx, axis=i) if Fx.ndim > i else Fx[::-1]
    if len(axes) == 1:
        Fx = Fx.ravel()
    epsilons = np.array([np.min(np.abs(np.diff(a))) for a in axes])/2
    assert epsilons.min() > 1e-18, "Difference between points must be larger than 2e-18"
    F = _GriddedInterpolant(axes, Fx, extrap)
    return (lambda *args: _interp_table_nd(F, epsilons, *args)), F, epsilons


class _GriddedInterpolant:
    """MATLAB's `griddedInterpolant(x, Fx, 'linear', extrap)`."""

    def __init__(self, axes, Fx, extrap):
        self.axes, self.Fx = axes, Fx
        self.ExtrapolationMethod = extrap

    def __call__(self, *queries):
        q = np.broadcast_arrays(*[np.asarray(v, dtype=float) for v in queries])
        shape = q[0].shape
        pts = np.column_stack([v.ravel() for v in q])
        method = self.ExtrapolationMethod
        if method == "nearest":
            clipped = np.column_stack([np.clip(pts[:, k], a[0], a[-1]) for k, a in enumerate(self.axes)])
            out = RegularGridInterpolator(tuple(self.axes), self.Fx, method="linear")(clipped)
        else:
            fill = np.nan if method == "none" else None
            out = RegularGridInterpolator(tuple(self.axes), self.Fx, method="linear",
                                          bounds_error=False, fill_value=fill)(pts)
        return out.reshape(shape)


def _interp_table_nd(Y: _GriddedInterpolant, epsilon: np.ndarray, *Yq: Any) -> Any:
    from pymrst.modules.ad_core.utils.double_to_adi import double_to_adi
    from pymrst.utils.adi import value
    assert len(Yq) == epsilon.size
    isad = [hasattr(v, "jac") for v in Yq]
    if not any(isad):
        return Y(*Yq)
    ad = next(v for v, a in zip(Yq, isad) if a)
    Y.ExtrapolationMethod = "linear"
    Yqd = [np.asarray(value(v), dtype=float) for v in Yq]
    Yev = Y(*Yqd)
    Ye = double_to_adi(np.ravel(Yev), ad)
    for i, a in enumerate(isad):
        if a:
            permuted = list(Yqd)
            permuted[i] = permuted[i] + epsilon[i]
            dydx = (np.ravel(Y(*permuted)) - np.ravel(Yev))/epsilon[i]
            Ye = Ye + dydx*(Yq[i] - np.ravel(Yqd[i]))
    return Ye


getMultiDimInterpolator = get_multi_dim_interpolator
__all__ = ["getMultiDimInterpolator", "get_multi_dim_interpolator"]


if __name__ == "__main__":
    from pymrst.utils.adi import init_variables_adi

    X, Yax = np.array([0.0, 1.0, 2.0]), np.array([10.0, 20.0])
    TABLE = np.array([[0.0, 1.0], [2.0, 3.0], [4.0, 7.0]])

    def test_values_extrapolation_and_reversed_axes():
        fn, _, eps = get_multi_dim_interpolator([X, Yax], TABLE)
        assert np.allclose(fn(np.array([0.5, 3.0]), np.array([15.0, 20.0])), [1.5, 11.0])
        assert np.allclose(eps, [0.5, 5.0])
        fn_rev, _, _ = get_multi_dim_interpolator([X[::-1], Yax], TABLE[::-1])
        assert np.allclose(fn_rev(np.array([0.5]), np.array([15.0])), [1.5])
        fn_none, _, _ = get_multi_dim_interpolator([X, Yax], TABLE, "none")
        assert np.isnan(fn_none(np.array([3.0]), np.array([15.0])))[0]
        print("ok test_values_extrapolation_and_reversed_axes")

    def test_ad_derivatives_by_half_spacing_differences():
        fn, _, _ = get_multi_dim_interpolator([X, Yax], TABLE, "none")
        p, q = init_variables_adi(np.array([0.5]), np.array([12.0]))
        out = fn(p, q)
        assert np.allclose(out.val, [0.5*2 + 0.2*1])
        assert np.allclose(out.jac[0].toarray(), [[2.0]]) and np.allclose(out.jac[1].toarray(), [[0.1]])
        print("ok test_ad_derivatives_by_half_spacing_differences")

    test_values_extrapolation_and_reversed_axes()
    test_ad_derivatives_by_half_spacing_differences()
    print("\nAll tests passed!")
