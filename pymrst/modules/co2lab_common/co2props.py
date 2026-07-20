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
