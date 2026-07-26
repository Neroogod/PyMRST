"""VEM utility layer (MRST ``vem/utils``): quadrature and polynomial integration."""

from pymrst.modules.vem.utils.quadrature import (
    poly_dim,
    polygon_int,
    polyhedron_int,
    tetrahedron_quad_rule,
    triangle_quad_rule,
)

__all__ = [
    "poly_dim",
    "polygon_int",
    "polyhedron_int",
    "tetrahedron_quad_rule",
    "triangle_quad_rule",
]
