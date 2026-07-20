"""MRST ``initCoreyFluid.m`` Corey incompressible fluid constructor.

PyMRST's validated `init_simple_fluid` already builds Corey relperms.
"""

from typing import Sequence

from pymrst.modules.incomp.fluid import Fluid
from pymrst.modules.incomp.fluid.incompressible.init_simple_fluid import (
    init_simple_fluid,
)


def init_corey_fluid(
    mu: Sequence[float],
    rho: Sequence[float],
    n: Sequence[float] = (1.0, 1.0),
) -> Fluid:
    """Construct the Corey fluid used by MRST's incompressible examples."""
    return init_simple_fluid(mu=mu, rho=rho, n=n)

__all__ = ["Fluid", "init_corey_fluid"]
