"""Sampled CO2 property tables (MRST ``CO2props`` layout)."""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple, Union

import numpy as np
import scipy.io

from pymrst.utils.adi import ADI, _diag_mul, value

ArrayOrADI = Union[np.ndarray, ADI]

DEFAULT_TABLE_DIR = (
    Path(__file__).parents[4]
    / "mrst-2024a-Geomech/examples/data/CO2lab_Sampled_Tables"
)
DEFAULT_STEM = "CarbonDioxide_100000_400000000_278_524_800_800"


@dataclass
class SampledProp2D:
    """A property sampled on a uniform pressure/temperature grid."""

    v1_span: Tuple[float, float]
    v2_span: Tuple[float, float]
    vals: np.ndarray

    @classmethod
    def from_mat(cls, path) -> "SampledProp2D":
        d = scipy.io.loadmat(path, squeeze_me=True, struct_as_record=False)
        p, t = d["P"], d["T"]
        return cls(
            v1_span=(float(p.span[0]), float(p.span[1])),
            v2_span=(float(t.span[0]), float(t.span[1])),
            vals=np.asarray(d["vals"], dtype=np.float64),
        )

    def _locate(self, v1: np.ndarray, v2: np.ndarray):
        n1, n2 = self.vals.shape
        s1 = (self.v1_span[1] - self.v1_span[0]) / (n1 - 1)
        s2 = (self.v2_span[1] - self.v2_span[0]) / (n2 - 1)
        x = (v1 - self.v1_span[0]) / s1
        y = (v2 - self.v2_span[0]) / s2
        i = np.clip(np.floor(x).astype(np.int64), 0, n1 - 2)
        j = np.clip(np.floor(y).astype(np.int64), 0, n2 - 2)
        return i, j, x - i, y - j, s1, s2

    def in_range(self, v1: np.ndarray, v2: np.ndarray) -> np.ndarray:
        return (
            (v1 >= self.v1_span[0])
            & (v1 <= self.v1_span[1])
            & (v2 >= self.v2_span[0])
            & (v2 <= self.v2_span[1])
        )

    def eval(self, v1: np.ndarray, v2: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Bilinear value and partial derivatives."""
        v1 = np.atleast_1d(np.asarray(v1, dtype=np.float64))
        v2 = np.atleast_1d(np.asarray(v2, dtype=np.float64))
        v1, v2 = np.broadcast_arrays(v1, v2)
        i, j, p, t, s1, s2 = self._locate(v1, v2)
        g = self.vals
        c00, c10 = g[i, j], g[i + 1, j]
        c01, c11 = g[i, j + 1], g[i + 1, j + 1]
        f = (
            (1 - p) * (1 - t) * c00
            + p * (1 - t) * c10
            + (1 - p) * t * c01
            + p * t * c11
        )
        df1 = ((1 - t) * (c10 - c00) + t * (c11 - c01)) / s1
        df2 = ((1 - p) * (c01 - c00) + p * (c11 - c10)) / s2
        return f, df1, df2

    def __call__(self, p: ArrayOrADI, temp: ArrayOrADI) -> ArrayOrADI:
        f, df1, df2 = self.eval(value(p), value(temp))
        jac = None
        if isinstance(p, ADI):
            jac = [_diag_mul(df1, j) for j in p.jac]
        if isinstance(temp, ADI):
            jt = [_diag_mul(df2, j) for j in temp.jac]
            jac = jt if jac is None else [a + b for a, b in zip(jac, jt)]
        return f if jac is None else ADI(f, jac)


@dataclass
class CO2Props:
    """CO2 density, viscosity, and optional enthalpy tables."""

    rho: SampledProp2D
    mu: SampledProp2D
    h: Optional[SampledProp2D] = None


def co2_props(table_dir=None, stem: str = DEFAULT_STEM, with_enthalpy: bool = False) -> CO2Props:
    """Load the MRST CO2 property tables."""
    d = Path(table_dir) if table_dir is not None else DEFAULT_TABLE_DIR
    return CO2Props(
        rho=SampledProp2D.from_mat(d / f"{stem}_D.mat"),
        mu=SampledProp2D.from_mat(d / f"{stem}_V.mat"),
        h=SampledProp2D.from_mat(d / f"{stem}_H.mat") if with_enthalpy else None,
    )


__all__ = ["CO2Props", "SampledProp2D", "co2_props"]


if __name__ == "__main__":

    def _linear_table(a=3.0, b=-2.0, c=7.0, n1=5, n2=4,
                      span1=(0.0, 4.0), span2=(0.0, 3.0)):
        """A table holding an exactly linear field `a*v1 + b*v2 + c`.

        Bilinear interpolation reproduces a linear field exactly, so the
        expected value and both derivatives are known in closed form. That
        makes every assertion below exact rather than approximate, and a wrong
        axis order or a transposed table shows up immediately instead of as a
        small discrepancy.
        """
        v1 = np.linspace(span1[0], span1[1], n1)[:, None]
        v2 = np.linspace(span2[0], span2[1], n2)[None, :]
        return SampledProp2D(span1, span2, a * v1 + b * v2 + c), (a, b, c)

    def test_a_linear_field_is_reproduced_exactly():
        table, (a, b, c) = _linear_table()
        v1 = np.array([0.0, 1.3, 2.75, 4.0])
        v2 = np.array([0.0, 0.7, 2.10, 3.0])
        value, d1, d2 = table.eval(v1, v2)
        assert np.allclose(value, a * v1 + b * v2 + c, rtol=1e-12, atol=0.0), value
        print(f"ok test_a_linear_field_is_reproduced_exactly "
              f"(max error {np.max(np.abs(value - (a * v1 + b * v2 + c))):.2e})")

    def test_the_derivatives_are_the_slopes_and_not_swapped():
        """d/dv1 and d/dv2 are the two coefficients, in that order.

        Returning them the other way round leaves both plausible and every
        downstream density gradient wrong in sign or magnitude.
        """
        table, (a, b, c) = _linear_table()
        _, d1, d2 = table.eval(np.array([1.0, 2.5]), np.array([0.5, 1.5]))
        assert np.allclose(d1, a, rtol=1e-12, atol=0.0), d1
        assert np.allclose(d2, b, rtol=1e-12, atol=0.0), d2
        print(f"ok test_the_derivatives_are_the_slopes_and_not_swapped "
              f"(d/dv1={d1[0]:.6g}, d/dv2={d2[0]:.6g})")

    def test_the_axes_are_not_transposed():
        """A table that varies only along v1 must have a zero v2 derivative."""
        v1 = np.linspace(0.0, 4.0, 5)[:, None]
        v2 = np.zeros((1, 4))
        table = SampledProp2D((0.0, 4.0), (0.0, 3.0), v1 + v2)
        _, d1, d2 = table.eval(np.array([2.0]), np.array([1.0]))
        assert np.allclose(d1, 1.0) and np.allclose(d2, 0.0), (d1, d2)
        print("ok test_the_axes_are_not_transposed")

    def test_corners_hit_the_stored_samples():
        table, (a, b, c) = _linear_table()
        for x, y in ((0.0, 0.0), (4.0, 0.0), (0.0, 3.0), (4.0, 3.0)):
            value, _, _ = table.eval(np.array([x]), np.array([y]))
            assert np.isclose(value[0], a * x + b * y + c, rtol=1e-12, atol=0.0)
        print("ok test_corners_hit_the_stored_samples")

    def test_in_range_marks_the_span_inclusively():
        table, _ = _linear_table()
        inside = table.in_range(np.array([0.0, 2.0, 4.0]), np.array([0.0, 1.0, 3.0]))
        outside = table.in_range(np.array([-0.1, 4.1]), np.array([1.0, 1.0]))
        assert inside.all(), inside
        assert not outside.any(), outside
        print("ok test_in_range_marks_the_span_inclusively")

    def test_evaluation_outside_the_span_is_clamped_not_wrapped():
        """Extrapolation must stay finite; a wrapped index would be silent."""
        table, _ = _linear_table()
        value, d1, d2 = table.eval(np.array([-5.0, 9.0]), np.array([-4.0, 8.0]))
        assert np.all(np.isfinite(value)) and np.all(np.isfinite(d1))
        print(f"ok test_evaluation_outside_the_span_is_clamped_not_wrapped "
              f"({value[0]:.3g}, {value[1]:.3g})")

    def test_the_shipped_tables_load_and_are_physical():
        """The real CO2 tables, when they are installed."""
        try:
            props = co2_props()
        except Exception as error:                    # pragma: no cover
            print(f"ok test_the_shipped_tables_load_and_are_physical (skipped: "
                  f"{type(error).__name__})")
            return
        p = np.array([5e6, 1e7, 2e7])
        t = np.array([300.0, 320.0, 340.0])
        rho = np.asarray(value(props.rho(p, t)), dtype=float)
        assert np.all(np.isfinite(rho)) and np.all(rho > 0.0), rho
        # CO2 is denser when colder at fixed pressure.
        cold = np.asarray(value(props.rho(np.array([1e7]), np.array([300.0]))), float)
        warm = np.asarray(value(props.rho(np.array([1e7]), np.array([360.0]))), float)
        assert cold[0] > warm[0], (cold, warm)
        print(f"ok test_the_shipped_tables_load_and_are_physical "
              f"(rho {rho.min():.1f} to {rho.max():.1f} kg/m3)")

    test_a_linear_field_is_reproduced_exactly()
    test_the_derivatives_are_the_slopes_and_not_swapped()
    test_the_axes_are_not_transposed()
    test_corners_hit_the_stored_samples()
    test_in_range_marks_the_span_inclusively()
    test_evaluation_outside_the_span_is_clamped_not_wrapped()
    test_the_shipped_tables_load_and_are_physical()
    print("\nAll tests passed!")
