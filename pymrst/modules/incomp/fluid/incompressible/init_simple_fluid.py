"""MRST ``initSimpleFluid.m`` two-phase incompressible Corey fluid."""

from typing import Sequence

import numpy as np

from pymrst.modules.incomp.fluid import Fluid


def init_simple_fluid(
    mu: Sequence[float],
    rho: Sequence[float],
    n: Sequence[float] = (1.0, 1.0),
) -> Fluid:
    """Two incompressible phases with Corey relperms ``s^n`` and ``(1-s)^n``."""
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

__all__ = ["Fluid", "init_simple_fluid"]
