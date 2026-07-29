"""DG discretization building blocks: polynomials and reference-element bases."""

from pymrst.modules.dg.discretizations.dg_basis import DGBasis, dg_basis
from pymrst.modules.dg.discretizations.dg_discretization import (
    DGDiscretization,
    compute_cell_dimensions,
)
from pymrst.modules.dg.discretizations.polynomials import (
    Polynomial,
    chebyshev_polynomials,
    legendre_polynomials,
    poly_dim,
    simple_polynomials,
    tensor_product,
)

__all__ = [
    "DGBasis",
    "dg_basis",
    "DGDiscretization",
    "compute_cell_dimensions",
    "Polynomial",
    "chebyshev_polynomials",
    "legendre_polynomials",
    "poly_dim",
    "simple_polynomials",
    "tensor_product",
]
