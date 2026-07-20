"""Rachford-Rice VLE solve (MRST ``solveRachfordRiceVLE.m`` layout)."""

import numpy as np


def rachford_rice(
    z: np.ndarray,
    K: np.ndarray,
    tol: float = 1e-12,
    max_it: int = 100,
) -> np.ndarray:
    """Vapour fraction solving the Rachford-Rice equation."""
    z = np.atleast_2d(np.asarray(z, float))
    K = np.atleast_2d(np.asarray(K, float))
    nc = z.shape[0]
    V = np.zeros(nc)
    for c in range(nc):
        k, zz = K[c], z[c]
        if k.max() <= 1.0 or k.min() >= 1.0:
            V[c] = 1.0 if k.min() >= 1.0 else 0.0
            continue
        lo = 1.0 / (1.0 - k.max()) + 1e-12
        hi = 1.0 / (1.0 - k.min()) - 1e-12
        lo, hi = max(lo, 0.0), min(hi, 1.0)
        f = lambda v: np.sum(zz * (k - 1.0) / (1.0 + v * (k - 1.0)))
        a, b = lo, hi
        fa, fb = f(a), f(b)
        if fa * fb > 0:
            V[c] = 1.0 if fa > 0 else 0.0
            continue
        for _ in range(max_it):
            m = 0.5 * (a + b)
            fm = f(m)
            if abs(fm) < tol or (b - a) < tol:
                break
            if fa * fm < 0:
                b, fb = m, fm
            else:
                a, fa = m, fm
        V[c] = 0.5 * (a + b)
    return V


def solveRachfordRiceVLE(z: np.ndarray, K: np.ndarray, tol: float = 1e-12, max_it: int = 100) -> np.ndarray:
    """MRST camelCase entry point for the Rachford-Rice VLE solve."""
    return rachford_rice(z, K, tol=tol, max_it=max_it)


def solve_rachford_rice_vle(z: np.ndarray, K: np.ndarray, tol: float = 1e-12, max_it: int = 100) -> np.ndarray:
    """Pythonic alias for :func:`solveRachfordRiceVLE`."""
    return solveRachfordRiceVLE(z, K, tol=tol, max_it=max_it)


__all__ = ["rachford_rice", "solveRachfordRiceVLE", "solve_rachford_rice_vle"]
