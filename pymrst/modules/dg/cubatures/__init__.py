"""DG cubatures: reference-element rules and their maps onto grid elements."""

from pymrst.modules.dg.cubatures.reference_cubature import (
    get_cube_cubature,
    get_line_cubature,
    get_square_cubature,
    get_tetrahedron_cubature,
    get_triangle_cubature,
)
from pymrst.modules.dg.cubatures.element_cubature import (
    ElementCubature,
    LineCubature,
    TetrahedronCubature,
    TriangleCubature,
)
from pymrst.modules.dg.cubatures.moment_fitting import (
    fit_moments,
    moment_fitting_cubature,
)

__all__ = [
    "get_line_cubature",
    "get_square_cubature",
    "get_triangle_cubature",
    "get_cube_cubature",
    "get_tetrahedron_cubature",
    "ElementCubature",
    "TriangleCubature",
    "TetrahedronCubature",
    "LineCubature",
    "fit_moments",
    "moment_fitting_cubature",
]
