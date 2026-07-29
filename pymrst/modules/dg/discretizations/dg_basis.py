"""DG basis functions on the reference element (MRST ``dgBasis``).

Builds a set of multivariate basis polynomials of total degree <= `degree` in
`dim` dimensions, as tensor products of a chosen 1-D family (`poly`, `legendre`
or `chebyshev`). The degrees of freedom are ordered by ascending total degree,
matching MRST so the dof numbering lines up.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np

from pymrst.modules.dg.discretizations.polynomials import (
    Polynomial,
    chebyshev_polynomials,
    legendre_polynomials,
    poly_dim,
    simple_polynomials,
    tensor_product,
)

_FAMILIES = {
    "poly": simple_polynomials,
    "legendre": legendre_polynomials,
    "chebyshev": chebyshev_polynomials,
}


@dataclass
class DGBasis:
    """A DG basis: polynomials `psi`, their gradients, multi-indices and count."""

    psi: List[Polynomial]
    grad_psi: List[List[Polynomial]]
    k: np.ndarray
    n_dof: int
    degree: int
    type: str

    def evaluate(self, x: np.ndarray) -> np.ndarray:
        """Values of every basis function at points `x`; returns `(n, n_dof)`."""
        x = np.atleast_2d(np.asarray(x, dtype=float))
        return np.column_stack([p(x) for p in self.psi])

    def evaluate_gradient(self, x: np.ndarray) -> np.ndarray:
        """Gradients at `x`; returns `(n, n_dof, dim)`."""
        x = np.atleast_2d(np.asarray(x, dtype=float))
        dim = self.k.shape[1]
        out = np.zeros((x.shape[0], self.n_dof, dim))
        for j, gp in enumerate(self.grad_psi):
            for d in range(dim):
                out[:, j, d] = gp[d](x)
        return out


def _multi_indices(degree, dim: int) -> np.ndarray:
    """Exponent rows of total degree <= max(degree), ordered as MRST's dgBasis."""
    degree = np.atleast_1d(np.asarray(degree, dtype=np.int64))
    max_degree = int(degree.max())
    m = poly_dim(max_degree, 1)                        # = max_degree + 1
    r = np.arange(m ** dim)
    k = np.column_stack([(r // m ** d) % m for d in range(dim)])
    k = k[k.sum(axis=1) <= max_degree]
    k = k[np.argsort(k.sum(axis=1), kind="stable")]    # ascending total degree
    if degree.size > 1:                                # per-dimension degree caps
        keep = np.all(k <= degree[None, :], axis=1)
        k = k[keep]
    return k


def dg_basis(dim: int, degree, type: str = "legendre") -> DGBasis:
    """Return the DG basis of the given `dim` and `degree` (MRST `dgBasis`).

    The argument order matches MRST: `dim` (number of dimensions) then `degree`
    (scalar total degree or a per-dimension array); `type` selects the 1-D
    family. Basis function `dofNo` is the tensor product of the 1-D polynomials
    at multi-index `k[dofNo]`.
    """
    if type not in _FAMILIES:
        raise ValueError(f"unknown basis type {type!r}")
    k = _multi_indices(degree, dim)
    max_degree = int(k.sum(axis=1).max()) if k.size else 0
    poly = _FAMILIES[type](max_degree)
    psi, grad_psi = [], []
    for row in k:
        p = tensor_product(*[poly[row[d]] for d in range(dim)])
        psi.append(p)
        grad_psi.append(p.grad())
    return DGBasis(psi=psi, grad_psi=grad_psi, k=k, n_dof=k.shape[0],
                   degree=max_degree, type=type)


if __name__ == "__main__":

    def test_ndof_matches_poly_dim():
        for dim in (1, 2, 3):
            for deg in range(4):
                b = dg_basis(dim, deg)
                assert b.n_dof == poly_dim(deg, dim), (dim, deg)
        print("ok test_ndof_matches_poly_dim")

    def test_multi_index_ordering_2d():
        # Matches MRST: ascending total degree, dim-0 exponent leading within a
        # degree.
        k = _multi_indices(2, 2)
        expected = np.array([[0, 0], [1, 0], [0, 1], [2, 0], [1, 1], [0, 2]])
        assert np.array_equal(k, expected), k
        print("ok test_multi_index_ordering_2d")

    def test_first_dof_is_constant():
        b = dg_basis(2, 3, "legendre")
        pts = np.random.default_rng(0).random((5, 2))
        assert np.allclose(b.psi[0](pts), 1.0)
        print("ok test_first_dof_is_constant")

    def test_gradient_of_legendre_tensor():
        # basis x*y-ish term: check gradient against finite differences.
        b = dg_basis(2, 2, "legendre")
        rng = np.random.default_rng(1)
        x = rng.random((4, 2))
        h = 1e-6
        g = b.evaluate_gradient(x)
        for d in range(2):
            xp, xm = x.copy(), x.copy()
            xp[:, d] += h
            xm[:, d] -= h
            fd = (b.evaluate(xp) - b.evaluate(xm)) / (2 * h)
            assert np.allclose(g[:, :, d], fd, atol=1e-6)
        print("ok test_gradient_of_legendre_tensor")

    def test_legendre_orthogonality_on_reference_square():
        # Tensor-product Legendre on [-1,1]^2 is orthogonal.
        from numpy.polynomial.legendre import leggauss
        b = dg_basis(2, 2, "legendre")
        gx, gw = leggauss(5)
        X, Y = np.meshgrid(gx, gx)
        W = np.outer(gw, gw).ravel()
        pts = np.column_stack([X.ravel(), Y.ravel()])
        V = b.evaluate(pts)
        M = V.T @ (W[:, None] * V)
        off = M - np.diag(np.diag(M))
        assert np.max(np.abs(off)) < 1e-12, np.max(np.abs(off))
        print("ok test_legendre_orthogonality_on_reference_square")

    test_ndof_matches_poly_dim()
    test_multi_index_ordering_2d()
    test_first_dof_is_constant()
    test_gradient_of_legendre_tensor()
    test_legendre_orthogonality_on_reference_square()
    print("\nAll tests passed!")
