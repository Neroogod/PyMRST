"""Bilinear interpolation on a regular grid (MRST ``interpRegular2DTable.m``).

The same bilinear interpolation as `interp_2d_table`, but for a table sampled
on a *regular* grid. Because the spacing is constant the bracketing cell is
found by arithmetic rather than a search, which is what makes this worth having
separately - it is called per cell per Newton iteration.

NaN queries propagate as NaN rather than raising, matching MRST, and queries
beyond the table clamp into the end cell so the bilinear form still applies.
"""

from dataclasses import dataclass
from typing import Tuple

import numpy as np

from pymrst.utils.adi import ADI, _diag_mul, value


@dataclass
class RegularTable2D:
    """A table on a regular grid: ``data`` sampled from ``(x0, y0)`` by (dx, dy)."""

    x0: float
    y0: float
    dx: float
    dy: float
    data: np.ndarray

    def __post_init__(self):
        self.data = np.asarray(self.data, dtype=float)
        if self.data.ndim != 2:
            raise ValueError(f"table data must be 2-D, got {self.data.shape}")
        if self.dx <= 0 or self.dy <= 0:
            raise ValueError("a regular table needs positive spacing")

    @property
    def x(self) -> np.ndarray:
        return self.x0 + self.dx * np.arange(self.data.shape[0])

    @property
    def y(self) -> np.ndarray:
        return self.y0 + self.dy * np.arange(self.data.shape[1])


def _corners(table: RegularTable2D, xi: np.ndarray, vi: np.ndarray):
    """The four bracketing samples and the local coordinates within the cell."""
    steps_x = table.data.shape[0] - 1
    steps_y = table.data.shape[1] - 1

    ix = np.clip(np.floor((xi - table.x0) / table.dx).astype(np.int64),
                 0, max(steps_x - 1, 0))
    iy = np.clip(np.floor((vi - table.y0) / table.dy).astype(np.int64),
                 0, max(steps_y - 1, 0))

    p = (xi - (table.x0 + ix * table.dx)) / table.dx
    t = (vi - (table.y0 + iy * table.dy)) / table.dy

    corners = np.column_stack([
        table.data[ix, iy], table.data[ix + 1, iy],
        table.data[ix, iy + 1], table.data[ix + 1, iy + 1],
    ])
    return corners, p, t


def interp_regular_2d_table(table: RegularTable2D, x, y):
    """Interpolate ``table`` at ``(x, y)``, transparent to ADI inputs."""
    xi = np.atleast_1d(value(x)).astype(float)
    vi = np.atleast_1d(value(y)).astype(float)
    missing = np.isnan(xi) | np.isnan(vi)

    values = np.full(xi.shape, np.nan)
    dx = np.full(xi.shape, np.nan)
    dv = np.full(xi.shape, np.nan)

    if np.any(~missing):
        corners, p, t = _corners(table, xi[~missing], vi[~missing])
        weight = np.column_stack([(1 - p) * (1 - t), p * (1 - t),
                                  (1 - p) * t, p * t])
        wx = np.column_stack([-(1 - t), (1 - t), -t, t])
        wy = np.column_stack([-(1 - p), -p, (1 - p), p])
        values[~missing] = np.sum(corners * weight, axis=1)
        dx[~missing] = np.sum(corners * wx, axis=1) / table.dx
        dv[~missing] = np.sum(corners * wy, axis=1) / table.dy

    if not isinstance(x, ADI) and not isinstance(y, ADI):
        return values
    jacobians = None
    if isinstance(x, ADI):
        jacobians = [_diag_mul(dx, j) for j in x.jac]
    if isinstance(y, ADI):
        other = [_diag_mul(dv, j) for j in y.jac]
        jacobians = other if jacobians is None else [
            a + b for a, b in zip(jacobians, other)]
    return ADI(values, jacobians)


interpRegular2DTable = interp_regular_2d_table


if __name__ == "__main__":

    def _bilinear():
        """data = x + 10*y on a unit grid, so answers are known exactly."""
        x = np.arange(3.0)
        y = np.arange(2.0)
        return RegularTable2D(x0=0.0, y0=0.0, dx=1.0, dy=1.0,
                              data=x[:, None] + 10.0 * y[None, :])

    def test_reproduces_the_table_at_its_nodes() -> None:
        table = _bilinear()
        for i, xv in enumerate(table.x):
            for j, yv in enumerate(table.y):
                out = interp_regular_2d_table(table, np.array([xv]),
                                              np.array([yv]))
                assert np.isclose(out[0], table.data[i, j]), (xv, yv, out)
        print("ok test_reproduces_the_table_at_its_nodes")

    def test_bilinear_field_is_exact_inside_the_cells() -> None:
        table = _bilinear()
        xi = np.array([0.5, 1.5, 0.25])
        vi = np.array([0.5, 0.25, 0.75])
        out = interp_regular_2d_table(table, xi, vi)
        assert np.allclose(out, xi + 10.0 * vi), out
        print("ok test_bilinear_field_is_exact_inside_the_cells")

    def test_agrees_with_the_irregular_implementation() -> None:
        """The regular form is an optimisation, so it must give the same answer."""
        from pymrst.modules.ad_props.utils.interp_2d_table import (
            Table2D, interp_2d_table,
        )
        table = _bilinear()
        general = Table2D(x=table.x, y=table.y, data=table.data)
        xi = np.array([0.3, 1.1, 1.9])
        vi = np.array([0.2, 0.8, 0.5])
        assert np.allclose(interp_regular_2d_table(table, xi, vi),
                           interp_2d_table(general, xi, vi))
        print("ok test_agrees_with_the_irregular_implementation")

    def test_carries_both_derivatives_through_adi() -> None:
        from pymrst.utils.adi import init_variables_adi
        table = _bilinear()
        p, t = init_variables_adi(np.array([0.5]), np.array([0.5]))
        out = interp_regular_2d_table(table, p, t)
        assert isinstance(out, ADI)
        assert np.isclose(out.jac[0].toarray()[0, 0], 1.0)
        assert np.isclose(out.jac[1].toarray()[0, 0], 10.0)
        print("ok test_carries_both_derivatives_through_adi")

    def test_nan_queries_propagate_rather_than_raise() -> None:
        table = _bilinear()
        out = interp_regular_2d_table(table, np.array([0.5, np.nan]),
                                      np.array([0.5, 0.5]))
        assert np.isclose(out[0], 5.5) and np.isnan(out[1]), out
        print("ok test_nan_queries_propagate_rather_than_raise")

    def test_queries_beyond_the_table_clamp_into_the_end_cell() -> None:
        table = _bilinear()
        out = interp_regular_2d_table(table, np.array([5.0]), np.array([0.0]))
        assert np.all(np.isfinite(out))
        print("ok test_queries_beyond_the_table_clamp_into_the_end_cell")

    def test_non_positive_spacing_is_rejected() -> None:
        try:
            RegularTable2D(x0=0.0, y0=0.0, dx=0.0, dy=1.0, data=np.zeros((2, 2)))
        except ValueError:
            print("ok test_non_positive_spacing_is_rejected")
            return
        raise AssertionError("zero spacing must be rejected")

    test_reproduces_the_table_at_its_nodes()
    test_bilinear_field_is_exact_inside_the_cells()
    test_agrees_with_the_irregular_implementation()
    test_carries_both_derivatives_through_adi()
    test_nan_queries_propagate_rather_than_raise()
    test_queries_beyond_the_table_clamp_into_the_end_cell()
    test_non_positive_spacing_is_rejected()
    print("\nAll tests passed!")
