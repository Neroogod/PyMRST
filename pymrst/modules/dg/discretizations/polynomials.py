"""Multivariate polynomials for DG basis functions (MRST ``dg`` module).

Ports MRST's `Polynomial` class and the 1-D `simple`/`legendre`/`chebyshev`
polynomial families. A polynomial is stored as a sum of terms, each a monomial
`prod(x**k)` scaled by a weight `w`; arithmetic combines terms and evaluation is
`sum_j w_j prod(x**k_j)`. These are the building blocks the DG basis
(`dg_basis`) tensor-products into multi-dimensional basis functions.
"""

from __future__ import annotations

from typing import List

import numpy as np


class Polynomial:
    """A multivariate polynomial as exponent rows `k` and term weights `w`.

    `k` has shape `(nterms, dim)` and `w` shape `(nterms,)`; the value at points
    `x` (shape `(n, dim)`) is `sum_j w_j * prod(x**k_j)`.
    """

    def __init__(self, k, w=None):
        k = np.atleast_2d(np.asarray(k, dtype=np.int64))
        if w is None:
            w = np.ones(k.shape[0])
        self.k = k
        self.w = np.asarray(w, dtype=float).ravel()
        self.dim = k.shape[1]

    def __call__(self, x: np.ndarray) -> np.ndarray:
        """Evaluate at points `x` of shape `(n, dim)`; returns `(n,)`."""
        x = np.atleast_2d(np.asarray(x, dtype=float))
        return (self.w[None, :] * np.prod(x[:, None, :] ** self.k[None, :, :],
                                          axis=2)).sum(axis=1)

    def dx(self, i: int) -> "Polynomial":
        """Partial derivative with respect to coordinate `i`."""
        w = self.w * self.k[:, i]
        k = self.k.copy()
        k[:, i] = np.maximum(k[:, i] - 1, 0)
        return Polynomial(k, w)

    def grad(self) -> List["Polynomial"]:
        """List of partial derivatives, one per dimension."""
        return [self.dx(i) for i in range(self.dim)]

    def __neg__(self) -> "Polynomial":
        return Polynomial(self.k, -self.w)

    def __add__(self, q) -> "Polynomial":
        if not isinstance(q, Polynomial):
            q = Polynomial(np.zeros((1, self.dim), dtype=np.int64), [float(q)])
        return Polynomial(np.vstack([self.k, q.k]), np.concatenate([self.w, q.w]))

    __radd__ = __add__

    def __sub__(self, q) -> "Polynomial":
        return self + (-q if isinstance(q, Polynomial) else -float(q))

    def __mul__(self, q) -> "Polynomial":
        if not isinstance(q, Polynomial):
            return Polynomial(self.k, self.w * float(q))
        # Outer product of terms: exponents add, weights multiply.
        pk = np.repeat(self.k, q.k.shape[0], axis=0)
        qk = np.tile(q.k, (self.k.shape[0], 1))
        pw = np.repeat(self.w, q.w.shape[0])
        qw = np.tile(q.w, self.w.shape[0])
        return Polynomial(pk + qk, pw * qw)

    __rmul__ = __mul__

    def __truediv__(self, scalar) -> "Polynomial":
        return Polynomial(self.k, self.w / float(scalar))


def tensor_product(*polys: Polynomial) -> Polynomial:
    """Combine 1-D polynomials into a multi-dimensional one (MRST tensorProduct).

    Argument `pNo` contributes exponents along dimension `pNo`; the result is
    their product in the multi-dimensional monomial basis.
    """
    pdim = len(polys)
    r = Polynomial(np.zeros((1, pdim), dtype=np.int64), [1.0])
    for p_no, p in enumerate(polys):
        pk = np.zeros((p.w.size, pdim), dtype=np.int64)
        pk[:, p_no] = p.k[:, 0]
        r = r * Polynomial(pk, p.w)
    return r


def poly_dim(k: int, dim: int) -> int:
    """Number of monomials of total degree <= k in `dim` variables."""
    if k < 0:
        return 0
    from math import comb
    return comb(k + dim, k)


def simple_polynomials(degree: int) -> List[Polynomial]:
    """1-D monomials `1, x, x^2, ...` up to `degree` (MRST simplePolynomials)."""
    return [Polynomial([[i]], [1.0]) for i in range(poly_dim(degree, 1))]


def legendre_polynomials(degree: int) -> List[Polynomial]:
    """1-D Legendre polynomials up to `degree` via the three-term recurrence."""
    n = poly_dim(degree, 1)
    p = [Polynomial([[0]], [1.0])]
    if degree > 0:
        p.append(Polynomial([[1]], [1.0]))
        for k in range(1, n - 1):
            p.append(((2 * k + 1) * p[1] * p[k] - k * p[k - 1]) / (k + 1))
    return p


def chebyshev_polynomials(degree: int) -> List[Polynomial]:
    """1-D Chebyshev polynomials (first kind) up to `degree`."""
    n = poly_dim(degree, 1)
    x = Polynomial([[1]], [1.0])
    t = [Polynomial([[0]], [1.0])]
    if degree > 0:
        t.append(Polynomial([[1]], [1.0]))
        for k in range(1, n - 1):
            t.append(2 * x * t[k] - t[k - 1])
    return t


if __name__ == "__main__":

    def test_evaluate_and_derivative():
        # p(x,y) = 3 x^2 y + 2 y
        p = Polynomial([[2, 1], [0, 1]], [3.0, 2.0])
        pts = np.array([[2.0, 1.0], [1.0, 3.0]])
        assert np.allclose(p(pts), [3 * 4 * 1 + 2 * 1, 3 * 1 * 3 + 2 * 3])
        dpx = p.dx(0)                                   # 6 x y
        assert np.allclose(dpx(pts), [6 * 2 * 1, 6 * 1 * 3])
        print("ok test_evaluate_and_derivative")

    def test_arithmetic():
        x = Polynomial([[1]], [1.0])
        q = (x + 1.0) * (x - 1.0)                       # x^2 - 1
        pts = np.array([[0.0], [2.0], [3.0]])
        assert np.allclose(q(pts), [-1.0, 3.0, 8.0])
        print("ok test_arithmetic")

    def test_legendre_recurrence():
        # Legendre: P0=1, P1=x, P2=(3x^2-1)/2, P3=(5x^3-3x)/2.
        L = legendre_polynomials(3)
        pts = np.array([[-1.0], [-0.3], [0.5], [1.0]])
        x = pts[:, 0]
        assert np.allclose(L[0](pts), 1.0)
        assert np.allclose(L[1](pts), x)
        assert np.allclose(L[2](pts), (3 * x ** 2 - 1) / 2)
        assert np.allclose(L[3](pts), (5 * x ** 3 - 3 * x) / 2)
        # Orthogonality on [-1, 1]: integral P2 P3 = 0.
        from numpy.polynomial.legendre import leggauss
        gx, gw = leggauss(6)
        gp = gx[:, None]
        assert abs((gw * L[2](gp) * L[3](gp)).sum()) < 1e-12
        print("ok test_legendre_recurrence")

    def test_chebyshev_recurrence():
        # Chebyshev: T0=1, T1=x, T2=2x^2-1, T3=4x^3-3x.
        T = chebyshev_polynomials(3)
        pts = np.array([[-0.7], [0.2], [0.9]])
        x = pts[:, 0]
        assert np.allclose(T[2](pts), 2 * x ** 2 - 1)
        assert np.allclose(T[3](pts), 4 * x ** 3 - 3 * x)
        print("ok test_chebyshev_recurrence")

    def test_tensor_product():
        L = legendre_polynomials(1)                     # [1, x] in 1-D
        psi = tensor_product(L[1], L[1])                # x*y
        pts = np.array([[2.0, 3.0], [1.0, 4.0]])
        assert np.allclose(psi(pts), [6.0, 4.0])
        assert psi.dim == 2
        print("ok test_tensor_product")

    test_evaluate_and_derivative()
    test_arithmetic()
    test_legendre_recurrence()
    test_chebyshev_recurrence()
    test_tensor_product()
    print("\nAll tests passed!")
