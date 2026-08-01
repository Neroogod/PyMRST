"""Virtual element method for incompressible flow (MRST ``modules/vem``).

Two entry points, at MRST's own names: `compute_virtual_ip` builds the per-cell
inner products from a grid and a rock, and `incomp_vem` solves the pressure
system they define. `vem_poisson` is PyMRST's own - a self-contained
first-order Poisson solver with no MRST counterpart, useful for reading the
method before the general implementation.
"""

from pymrst.modules.vem.compute_virtual_ip import compute_virtual_ip
from pymrst.modules.vem.incomp_vem import incomp_vem
from pymrst.modules.vem.utils import (
    VemBC,
    add_bc_vem,
    compute_vem_geometry,
    conserve_flux,
    poly_dim,
    polygon_int,
    polygon_int_3d,
    polyhedron_int,
    squeeze_block_diag,
    tetrahedron_quad_rule,
    triangle_quad_rule,
)
from pymrst.modules.vem.vem_poisson import vem_poisson

__all__ = [
    "VemBC",
    "add_bc_vem",
    "compute_virtual_ip",
    "incomp_vem",
    "compute_vem_geometry",
    "conserve_flux",
    "poly_dim",
    "polygon_int",
    "polygon_int_3d",
    "polyhedron_int",
    "squeeze_block_diag",
    "tetrahedron_quad_rule",
    "triangle_quad_rule",
    "vem_poisson",
]
