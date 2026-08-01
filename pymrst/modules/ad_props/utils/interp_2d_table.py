"""Bilinear tensor-table interpolation (MRST ``ad-props/utils/interp2DTable.m``).

A tensor table is a matrix `data[i, j]` sampled at `x[i]` and `y[j]`. A query
`(xi, vi)` is answered by interpolating along `x` in the two columns
bracketing `vi`, then blending those two results linearly in `y`.

Interpolation along `x` extrapolates linearly beyond the table, but the `y`
blend does not: `vi` outside the table's `y` range is clamped to its end
interval, matching MRST, which warns rather than refusing.
"""

from dataclasses import dataclass
from typing import Optional, Tuple, Union
import warnings

import numpy as np

from pymrst.utils.adi import ADI, _diag_mul, value
from pymrst.utils.interp_table import _interp1


@dataclass
class Table2D:
    """A tensor table: ``data`` sampled at ``x`` down rows and ``y`` across."""

    x: np.ndarray
    y: np.ndarray
    data: np.ndarray

    def __post_init__(self):
        self.x = np.asarray(self.x, dtype=float).ravel()
        self.y = np.asarray(self.y, dtype=float).ravel()
        self.data = np.asarray(self.data, dtype=float)
        if self.data.shape != (self.x.size, self.y.size):
            raise ValueError(
                f"table data is {self.data.shape}, expected "
                f"{(self.x.size, self.y.size)}"
            )


def _kernel(table: Table2D, xi: np.ndarray, vi: np.ndarray):
    """Values and both partial derivatives at ``(xi, vi)``."""
    y = table.y
    # the end intervals are open, so a query outside the table clamps into them
    bin_index = np.clip(np.searchsorted(y, vi, side="right") - 1,
                        0, max(y.size - 2, 0))
    span = y[bin_index + 1] - y[bin_index]
    weight = (vi - y[bin_index]) / span
    dweight = 1.0 / span

    left = np.zeros(xi.shape)
    right = np.zeros(xi.shape)
    dleft = np.zeros(xi.shape)
    dright = np.zeros(xi.shape)
    for column in range(y.size - 1):
        selected = bin_index == column
        if not np.any(selected):
            continue
        left[selected], dleft[selected] = _interp1(
            table.x, table.data[:, column], xi[selected])
        right[selected], dright[selected] = _interp1(
            table.x, table.data[:, column + 1], xi[selected])

    values = left * (1.0 - weight) + right * weight
    dx = dleft * (1.0 - weight) + dright * weight
    dv = -(left - right) * dweight
    return values, dx, dv


def interp_2d_table(table: Table2D, x, y, *, warn_out_of_range: bool = True):
    """Interpolate ``table`` at ``(x, y)``, transparent to ADI inputs."""
    xi = np.atleast_1d(value(x)).astype(float)
    vi = np.atleast_1d(value(y)).astype(float)

    if warn_out_of_range:
        out_x = xi.min() < table.x.min() or xi.max() > table.x.max()
        out_y = vi.min() < table.y.min() or vi.max() > table.y.max()
        if out_x or out_y:
            warnings.warn("value out of range of tensor table",
                          RuntimeWarning, stacklevel=2)

    values, dx, dv = _kernel(table, xi, vi)
    if not isinstance(x, ADI) and not isinstance(y, ADI):
        return values

    jacobians = None
    if isinstance(x, ADI):
        jacobians = [_diag_mul(dx, j) for j in x.jac]
    if isinstance(y, ADI):
        contribution = [_diag_mul(dv, j) for j in y.jac]
        jacobians = (contribution if jacobians is None
                     else [a + b for a, b in zip(jacobians, contribution)])
    return ADI(values, jacobians)


interp2DTable = interp_2d_table


if __name__ == "__main__":

    def _bilinear():
        """data = x + 10*y, so every interpolated value has a known answer."""
        x = np.array([0.0, 1.0, 2.0])
        y = np.array([0.0, 1.0])
        data = x[:, None] + 10.0 * y[None, :]
        return Table2D(x=x, y=y, data=data)

    def test_reproduces_the_table_at_its_nodes() -> None:
        table = _bilinear()
        for i, xv in enumerate(table.x):
            for j, yv in enumerate(table.y):
                out = interp_2d_table(table, np.array([xv]), np.array([yv]))
                assert np.isclose(out[0], table.data[i, j]), (xv, yv, out)
        print("ok test_reproduces_the_table_at_its_nodes")

    def test_bilinear_field_is_exact_between_nodes() -> None:
        table = _bilinear()
        xi = np.array([0.5, 1.5, 0.25])
        vi = np.array([0.5, 0.25, 0.75])
        out = interp_2d_table(table, xi, vi)
        assert np.allclose(out, xi + 10.0 * vi), out
        print("ok test_bilinear_field_is_exact_between_nodes")

    def test_partial_derivatives_match_the_field() -> None:
        table = _bilinear()
        _, dx, dv = _kernel(table, np.array([0.5]), np.array([0.5]))
        assert np.isclose(dx[0], 1.0), dx      # d(x + 10y)/dx
        assert np.isclose(dv[0], 10.0), dv     # d(x + 10y)/dy
        print("ok test_partial_derivatives_match_the_field")

    def test_carries_derivatives_from_either_argument() -> None:
        from pymrst.utils.adi import init_variables_adi
        table = _bilinear()
        p, r = init_variables_adi(np.array([0.5]), np.array([0.5]))
        out = interp_2d_table(table, p, r)
        assert isinstance(out, ADI)
        assert np.isclose(out.jac[0].toarray()[0, 0], 1.0)
        assert np.isclose(out.jac[1].toarray()[0, 0], 10.0)
        print("ok test_carries_derivatives_from_either_argument")

    def test_one_adi_argument_is_enough() -> None:
        from pymrst.utils.adi import init_variables_adi
        table = _bilinear()
        (p,) = init_variables_adi(np.array([0.5]))
        out = interp_2d_table(table, p, np.array([0.5]))
        assert isinstance(out, ADI) and len(out.jac) == 1
        print("ok test_one_adi_argument_is_enough")

    def test_out_of_range_warns_but_still_answers() -> None:
        table = _bilinear()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            out = interp_2d_table(table, np.array([5.0]), np.array([0.5]))
        assert any("out of range" in str(w.message) for w in caught)
        assert np.all(np.isfinite(out))
        print("ok test_out_of_range_warns_but_still_answers")

    def test_mismatched_table_shape_is_rejected() -> None:
        try:
            Table2D(x=np.zeros(3), y=np.zeros(2), data=np.zeros((2, 2)))
        except ValueError:
            print("ok test_mismatched_table_shape_is_rejected")
            return
        raise AssertionError("a data/axis shape mismatch must be rejected")

    test_reproduces_the_table_at_its_nodes()
    test_bilinear_field_is_exact_between_nodes()
    test_partial_derivatives_match_the_field()
    test_carries_derivatives_from_either_argument()
    test_one_adi_argument_is_enough()
    test_out_of_range_warns_but_still_answers()
    test_mismatched_table_shape_is_rejected()
    print("\nAll tests passed!")
