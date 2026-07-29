"""Moment-fitting cubatures for the DG module (MRST ``fitMoments`` / MomentFitting).

Compresses a rich candidate point set into a minimal non-negative quadrature that
still reproduces a target set of basis moments exactly. Given points `x`, a DG
`basis` and the moments `M_j = integral(psi_j)`, it solves the non-negative
least-squares `P w = M` (where `P[j,i] = psi_j(x_i)`) and then greedily drops the
points contributing least to the basis until the fit can no longer stay exact -
yielding a rule with at most `nDof` points that integrates every basis function
(hence every polynomial up to the basis degree) exactly.
"""

from typing import Tuple

import numpy as np
from scipy.optimize import nnls

from pymrst.modules.dg.discretizations.dg_basis import DGBasis, dg_basis
from pymrst.modules.dg.cubatures.reference_cubature import (
    get_cube_cubature,
    get_square_cubature,
)


def fit_moments(x: np.ndarray, basis: DGBasis, moments: np.ndarray,
                tol: float = 1e-10, reduce: bool = True
                ) -> Tuple[np.ndarray, np.ndarray]:
    """Fit a minimal non-negative rule at points `x` reproducing `moments`.

    Returns `(points, weights)`: a subset of `x` and weights `w >= 0` with
    `sum_i w_i psi_j(x_i) = moments_j` for every basis function `j`.
    """
    x = np.array(x, dtype=float)
    M = np.asarray(moments, dtype=float).ravel()

    def solve(pts):
        P = basis.evaluate(pts).T                      # (nDof, n)
        w, rnorm = nnls(P, M)
        return P, w, rnorm

    P, w, rnorm = solve(x)
    if not reduce:
        return x, w

    keep = np.ones(x.shape[0], dtype=bool)
    changed = True
    while changed:
        changed = False
        P = basis.evaluate(x[keep]).T
        order = np.argsort((P ** 2).sum(axis=0))       # least basis contribution
        idx = np.flatnonzero(keep)
        for local in order:
            trial = keep.copy()
            trial[idx[local]] = False
            _, w_try, rnorm_try = solve(x[trial])
            if rnorm_try < tol:                        # fit still exact
                keep = trial
                changed = True
                break

    _, w, _ = solve(x[keep])
    return x[keep], w


def _reference_candidates_and_moments(dim: int, precision: int):
    """Candidate points (a fine tensor-Gauss grid) and the exact reference-square
    /-cube Legendre moments `M_j = integral_ref psi_j`."""
    basis = dg_basis(dim, precision, "legendre")
    from numpy.polynomial.legendre import leggauss
    gx, _ = leggauss(precision + 2)
    grids = np.meshgrid(*([gx] * dim), indexing="ij")
    cand = np.column_stack([g.ravel() for g in grids])
    getter = get_square_cubature if dim == 2 else get_cube_cubature
    xm, wm, _, _ = getter(2 * precision + 1)
    measure = 2.0 ** dim
    M = measure * (basis.evaluate(xm).T @ wm)          # integral psi_j
    return basis, cand, M


def moment_fitting_cubature(dim: int, precision: int
                            ) -> Tuple[np.ndarray, np.ndarray]:
    """A compressed reference-element cubature of the given `precision`.

    Returns `(points, weights)` on [-1, 1]^dim (weights give the integral) that
    integrate every polynomial of total degree <= precision exactly, with at most
    `nDof` points - MRST's MomentFitting cubature on the reference element.
    """
    basis, cand, M = _reference_candidates_and_moments(dim, precision)
    return fit_moments(cand, basis, M)


if __name__ == "__main__":
    from itertools import product

    def test_fit_reproduces_moments_and_compresses():
        for dim in (2, 3):
            for p in (1, 2, 3):
                basis, cand, M = _reference_candidates_and_moments(dim, p)
                pts, w = fit_moments(cand, basis, M)
                # Reproduces every basis moment.
                assert np.allclose(basis.evaluate(pts).T @ w, M, atol=1e-9), (dim, p)
                # Minimal: at most nDof points.
                assert pts.shape[0] <= basis.n_dof, (dim, p, pts.shape[0])
                # Non-negative weights.
                assert np.all(w >= -1e-12)
        print("ok test_fit_reproduces_moments_and_compresses")

    def test_integrates_monomials_exactly():
        # The compressed rule integrates every monomial up to `precision` exactly.
        for dim in (2, 3):
            for p in (2, 3):
                pts, w = moment_fitting_cubature(dim, p)
                for exps in product(range(p + 1), repeat=dim):
                    if sum(exps) > p:
                        continue
                    num = float(w @ np.prod(pts ** np.array(exps), axis=1))
                    exact = 1.0
                    for e in exps:
                        exact *= (1 - (-1) ** (e + 1)) / (e + 1)
                    assert abs(num - exact) < 1e-9, (dim, p, exps)
        print("ok test_integrates_monomials_exactly")

    def test_weights_positive_and_measure():
        pts, w = moment_fitting_cubature(2, 3)
        assert np.all(w > 0)
        assert abs(w.sum() - 4.0) < 1e-9               # area of [-1,1]^2
        print(f"ok test_weights_positive_and_measure ({pts.shape[0]} pts)")

    test_fit_reproduces_moments_and_compresses()
    test_integrates_monomials_exactly()
    test_weights_positive_and_measure()
    print("\nAll tests passed!")
