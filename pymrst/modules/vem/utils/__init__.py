"""VEM utility layer (MRST ``vem/utils``).

One module per MRST function, at MRST's own names. This package re-exports
them so `from pymrst.modules.vem.utils import poly_dim` keeps working, which is
what let the split happen without touching a single call site.

The re-exports are a convenience, not an indirection: each symbol is defined in
the file named after its MRST counterpart, and that file is the one to read.
"""

from pymrst.modules.vem.utils.add_bc_vem import VemBC, add_bc_vem
from pymrst.modules.vem.utils.compute_vem_geometry import compute_vem_geometry
from pymrst.modules.vem.utils.conserve_flux import conserve_flux
from pymrst.modules.vem.utils.poly_dim import poly_dim
from pymrst.modules.vem.utils.polygon_int import polygon_int
from pymrst.modules.vem.utils.polygon_int_3d import polygon_int_3d
from pymrst.modules.vem.utils.polyhedron_int import polyhedron_int
from pymrst.modules.vem.utils.squeeze_block_diag import squeeze_block_diag
from pymrst.modules.vem.utils.tetrahedron_quad_rule import tetrahedron_quad_rule
from pymrst.modules.vem.utils.triangle_quad_rule import triangle_quad_rule

__all__ = [
    "VemBC",
    "add_bc_vem",
    "compute_vem_geometry",
    "conserve_flux",
    "poly_dim",
    "polygon_int",
    "polygon_int_3d",
    "polyhedron_int",
    "squeeze_block_diag",
    "tetrahedron_quad_rule",
    "triangle_quad_rule",
]
