"""Sampled CO2 property tables (MRST co2lab-common: SampledProp2D/CO2props).

Density, viscosity, and enthalpy of CO2 are tabulated on a uniform
(pressure, temperature) grid and interpolated bilinearly. The tables ship
with MRST as CarbonDioxide_<pmin>_<pmax>_<tmin>_<tmax>_<n>_<n>_{D,V,H}.mat.

Interpolation is AD-transparent: passing an ADI pressure returns an ADI
with the correct dRho/dp from the same bilinear stencil.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple, Union

import numpy as np
import scipy.io
import scipy.sparse as sp

from pymrst.utils.adi import ADI, _diag_mul, value

ArrayOrADI = Union[np.ndarray, ADI]

DEFAULT_TABLE_DIR = (
    Path(__file__).parents[4]
    / "mrst-2024a-Geomech/examples/data/CO2lab_Sampled_Tables"
)
DEFAULT_STEM = "CarbonDioxide_100000_400000000_278_524_800_800"


@dataclass
class SampledProp2D:
    """A property sampled on a uniform (v1, v2) grid, bilinearly interpolated."""

    v1_span: Tuple[float, float]
    v2_span: Tuple[float, float]
    vals: np.ndarray                 # (n1, n2)

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
        return ((v1 >= self.v1_span[0]) & (v1 <= self.v1_span[1])
                & (v2 >= self.v2_span[0]) & (v2 <= self.v2_span[1]))

    def eval(self, v1: np.ndarray, v2: np.ndarray
             ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Bilinear value and its partial derivatives (f, df/dv1, df/dv2)."""
        v1 = np.atleast_1d(np.asarray(v1, dtype=np.float64))
        v2 = np.atleast_1d(np.asarray(v2, dtype=np.float64))
        v1, v2 = np.broadcast_arrays(v1, v2)
        i, j, p, t, s1, s2 = self._locate(v1, v2)
        g = self.vals
        c00, c10 = g[i, j], g[i + 1, j]
        c01, c11 = g[i, j + 1], g[i + 1, j + 1]
        f = ((1 - p) * (1 - t) * c00 + p * (1 - t) * c10
             + (1 - p) * t * c01 + p * t * c11)
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
    """CO2 density (kg/m3) and viscosity (Pa s) from the sampled tables."""

    rho: SampledProp2D
    mu: SampledProp2D
    h: Optional[SampledProp2D] = None


def co2_props(table_dir=None, stem: str = DEFAULT_STEM,
              with_enthalpy: bool = False) -> CO2Props:
    """Load the MRST CO2 property tables (CO2props)."""
    d = Path(table_dir) if table_dir is not None else DEFAULT_TABLE_DIR
    return CO2Props(
        rho=SampledProp2D.from_mat(d / f"{stem}_D.mat"),
        mu=SampledProp2D.from_mat(d / f"{stem}_V.mat"),
        h=SampledProp2D.from_mat(d / f"{stem}_H.mat") if with_enthalpy
        else None,
    )


if __name__ == "__main__":
    from pymrst.utils.adi import init_variables_adi
    from pymrst.utils.units import BARSA

    co2 = co2_props()

    def test_table_loaded():
        """Tables have the documented range and resolution."""
        assert co2.rho.vals.shape == (800, 800)
        assert co2.rho.v1_span == (1e5, 4e8)
        assert co2.rho.v2_span == (278.0, 524.0)
        print("ok test_table_loaded")

    def test_density_physical():
        """CO2 density at storage conditions is supercritical (~700 kg/m3)."""
        p = np.array([100 * BARSA, 200 * BARSA, 300 * BARSA])
        t = np.full(3, 320.0)
        rho = co2.rho(p, t)
        assert np.all(rho > 200.0) and np.all(rho < 1100.0)
        assert np.all(np.diff(rho) > 0), "density must rise with pressure"
        mu = co2.mu(p, t)
        assert np.all(mu > 1e-5) and np.all(mu < 2e-4)
        print(f"ok test_density_physical (rho = {rho.round(1)} kg/m3)")

    def test_nodes_reproduced():
        """At table nodes the interpolant returns the tabulated value."""
        g = co2.rho
        n1, n2 = g.vals.shape
        s1 = (g.v1_span[1] - g.v1_span[0]) / (n1 - 1)
        s2 = (g.v2_span[1] - g.v2_span[0]) / (n2 - 1)
        for i, j in ((0, 0), (137, 402), (799, 799)):
            p = g.v1_span[0] + i * s1
            t = g.v2_span[0] + j * s2
            assert np.isclose(g(np.array([p]), np.array([t]))[0],
                              g.vals[i, j], rtol=1e-12)
        print("ok test_nodes_reproduced")

    def test_ad_derivative_matches_fd():
        """The ADI derivative equals a finite difference of the interpolant."""
        p0 = np.array([120 * BARSA, 250 * BARSA])
        t0 = np.full(2, 330.0)
        (p,) = init_variables_adi(p0)
        rho = co2.rho(p, t0)
        jac = rho.jac[0].toarray().diagonal()
        h = 1.0
        fd = (co2.rho(p0 + h, t0) - co2.rho(p0 - h, t0)) / (2 * h)
        assert np.allclose(jac, fd, rtol=1e-6), (jac, fd)
        print("ok test_ad_derivative_matches_fd")

    test_table_loaded()
    test_density_physical()
    test_nodes_reproduced()
    test_ad_derivative_matches_fd()
    print("\nAll tests passed!")
