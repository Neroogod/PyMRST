"""MRST ``initSingleFluid.m`` single-phase incompressible fluid."""

import numpy as np

from pymrst.modules.incomp.fluid import Fluid


def init_single_fluid(mu: float, rho: float) -> Fluid:
    """Single incompressible phase with viscosity ``mu`` and density ``rho``."""
    return Fluid(
        mu=np.array([float(mu)]),
        rho=np.array([float(rho)]),
        relperm=lambda s: np.atleast_2d(np.asarray(s, dtype=np.float64)),
        d_relperm=lambda s: np.ones_like(np.atleast_2d(s)),
    )

__all__ = ["Fluid", "init_single_fluid"]
