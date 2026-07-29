"""VEM utility layer (MRST ``vem/utils``): quadrature and polynomial integration."""

from pymrst.modules.vem.utils.quadrature import (
    poly_dim,
    polygon_int,
    polygon_int_3d,
    polyhedron_int,
    tetrahedron_quad_rule,
    triangle_quad_rule,
)
from pymrst.modules.vem.utils.vem_geometry import compute_vem_geometry

__all__ = [
    "poly_dim",
    "polygon_int",
    "polygon_int_3d",
    "polyhedron_int",
    "tetrahedron_quad_rule",
    "triangle_quad_rule",
    "compute_vem_geometry",
]
