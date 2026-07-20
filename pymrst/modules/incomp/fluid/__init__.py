"""Incompressible fluid objects, ported from MRST incomp/fluid.

A Fluid bundles constant viscosities and densities per phase with a
relative-permeability function kr(s) and, for transport solvers, its
derivative dkr(s). init_single_fluid mirrors initSingleFluid (kr = s);
init_simple_fluid mirrors initSimpleFluid (Corey power-law relperms
krW = s^n1, krO = (1-s)^n2).
"""

from dataclasses import dataclass
from typing import Callable, Optional, Sequence

import numpy as np


@dataclass
class Fluid:
    """Incompressible fluid: mu/rho per phase, kr(s), and optional dkr(s).

    relperm maps saturations (nc, n_phases) to kr (nc, n_phases).
    d_relperm maps them to [dkr_w/ds_w, dkr_o/ds_o] (nc, 2), each phase
    differentiated with respect to its own saturation (two-phase only).
    """

    mu: np.ndarray
    rho: np.ndarray
    relperm: Callable[[np.ndarray], np.ndarray]
    d_relperm: Optional[Callable[[np.ndarray], np.ndarray]] = None

    @property
    def n_phases(self) -> int:
        return self.mu.size

    def mobility(self, s: np.ndarray) -> np.ndarray:
        """Phase mobilities kr(s)/mu, shape (nc, n_phases)."""
        return self.relperm(s) / self.mu

    def d_mobility(self, s: np.ndarray) -> np.ndarray:
        """Mobility derivatives w.r.t. own saturation, shape (nc, 2)."""
        assert self.d_relperm is not None, "fluid has no relperm derivative"
        return self.d_relperm(s) / self.mu


def init_single_fluid(mu: float, rho: float) -> Fluid:
    """Single incompressible phase with viscosity mu (Pa*s), density rho."""
    return Fluid(
        mu=np.array([float(mu)]),
        rho=np.array([float(rho)]),
        relperm=lambda s: np.atleast_2d(np.asarray(s, dtype=np.float64)),
        d_relperm=lambda s: np.ones_like(np.atleast_2d(s)),
    )


def init_simple_fluid(
    mu: Sequence[float], rho: Sequence[float], n: Sequence[float] = (1.0, 1.0)
) -> Fluid:
    """Two incompressible phases with Corey relperms s^n[0], (1-s)^n[1]."""
    mu = np.asarray(mu, dtype=np.float64).ravel()
    rho = np.asarray(rho, dtype=np.float64).ravel()
    n = np.asarray(n, dtype=np.float64).ravel()
    assert mu.size == rho.size == n.size == 2, (
        "mu, rho, and Corey exponents must each have exactly two entries"
    )

    def relperm(s: np.ndarray) -> np.ndarray:
        s1 = np.atleast_2d(np.asarray(s, dtype=np.float64))[:, 0]
        return np.column_stack([s1 ** n[0], (1.0 - s1) ** n[1]])

    def d_relperm(s: np.ndarray) -> np.ndarray:
        s1 = np.atleast_2d(np.asarray(s, dtype=np.float64))[:, 0]
        return np.column_stack([
            n[0] * s1 ** (n[0] - 1.0),
            n[1] * (1.0 - s1) ** (n[1] - 1.0),
        ])

    return Fluid(mu=mu, rho=rho, relperm=relperm, d_relperm=d_relperm)


if __name__ == "__main__":
    from pymrst.utils.units import CENTI, POISE

    def test_single_fluid():
        """kr equals saturation; mobility is s/mu."""
        f = init_single_fluid(mu=1 * CENTI * POISE, rho=1000.0)
        assert f.n_phases == 1
        s = np.ones((5, 1))
        assert np.allclose(f.relperm(s), 1.0)
        assert np.allclose(f.mobility(s), 1.0 / 1e-3)
        print("ok test_single_fluid")

    def test_simple_fluid_corey():
        """Quadratic Corey relperms and derivatives are exact."""
        f = init_simple_fluid(mu=[1e-3, 10e-3], rho=[1000.0, 700.0],
                              n=[2.0, 2.0])
        s = np.array([[0.0, 1.0], [0.5, 0.5], [1.0, 0.0]])
        kr = f.relperm(s)
        assert np.allclose(kr[:, 0], [0.0, 0.25, 1.0])
        assert np.allclose(kr[:, 1], [1.0, 0.25, 0.0])
        dkr = f.d_relperm(s)
        assert np.allclose(dkr[:, 0], [0.0, 1.0, 2.0])   # 2s
        assert np.allclose(dkr[:, 1], [2.0, 1.0, 0.0])   # 2(1-s)
        print("ok test_simple_fluid_corey")

    def test_derivative_matches_finite_difference():
        """d_relperm agrees with a central difference of relperm."""
        f = init_simple_fluid(mu=[1e-3, 5e-3], rho=[1000.0, 800.0],
                              n=[3.0, 2.5])
        s1 = np.linspace(0.05, 0.95, 19)
        s = np.column_stack([s1, 1 - s1])
        h = 1e-7
        up = f.relperm(np.column_stack([s1 + h, 1 - s1 - h]))
        dn = f.relperm(np.column_stack([s1 - h, 1 - s1 + h]))
        fd = (up - dn) / (2 * h)
        dkr = f.d_relperm(s)
        assert np.allclose(dkr[:, 0], fd[:, 0], rtol=1e-6)
        assert np.allclose(dkr[:, 1], -fd[:, 1], rtol=1e-6)  # own-sat sign
        print("ok test_derivative_matches_finite_difference")

    test_single_fluid()
    test_simple_fluid_corey()
    test_derivative_matches_finite_difference()
    print("\nAll tests passed!")
