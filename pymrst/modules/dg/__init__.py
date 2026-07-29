"""Discontinuous Galerkin discretization (MRST ``modules/dg``).

The framework-free mathematical foundation is ported: the multivariate
`Polynomial` class, the 1-D `legendre`/`chebyshev`/`simple` families, and the
reference-element basis builder `dg_basis`. The higher layers (cubatures, the
`DGDiscretization` operators and the AD transport model) build on this.
"""

from pymrst.modules.dg.cubatures import (
    ElementCubature,
    LineCubature,
    TetrahedronCubature,
    TriangleCubature,
    fit_moments,
    get_cube_cubature,
    get_line_cubature,
    get_square_cubature,
    get_tetrahedron_cubature,
    get_triangle_cubature,
    moment_fitting_cubature,
)
from pymrst.modules.dg.discretizations import (
    DGBasis,
    DGDiscretization,
    Polynomial,
    chebyshev_polynomials,
    compute_cell_dimensions,
    dg_basis,
    legendre_polynomials,
    poly_dim,
    simple_polynomials,
    tensor_product,
)
from pymrst.modules.dg.dg_transport import DGScalarTransport

__all__ = [
    "DGBasis",
    "dg_basis",
    "Polynomial",
    "chebyshev_polynomials",
    "legendre_polynomials",
    "poly_dim",
    "simple_polynomials",
    "tensor_product",
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
    "DGDiscretization",
    "compute_cell_dimensions",
    "DGScalarTransport",
]
