"""Virtual element method for incompressible flow (MRST ``modules/vem``).

The utility foundation - triangle quadrature and polygonal polynomial
integration - is ported; the full `computeVirtualIP` local inner product builds
on it.
"""

from pymrst.modules.vem.utils.quadrature import (
    poly_dim,
    polygon_int,
    polyhedron_int,
    tetrahedron_quad_rule,
    triangle_quad_rule,
)
from pymrst.modules.vem.vem_poisson import vem_poisson
from pymrst.modules.vem.compute_virtual_ip import (
    compute_virtual_ip,
    incomp_vem,
)

__all__ = [
    "poly_dim",
    "polygon_int",
    "polyhedron_int",
    "tetrahedron_quad_rule",
    "triangle_quad_rule",
    "vem_poisson",
    "compute_virtual_ip",
    "incomp_vem",
]
