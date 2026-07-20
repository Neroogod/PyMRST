"""Vectorized cubic root solver (MRST ``mrstCubic.m`` layout)."""

from __future__ import annotations

import numpy as np


def _as_vector(x) -> np.ndarray:
    return np.atleast_1d(np.asarray(x))


def _complex_cuberoot(x: np.ndarray) -> np.ndarray:
    return np.power(np.asarray(x, dtype=complex), 1.0 / 3.0)


def mrstCubic(a, b=None, c=None, d=None) -> np.ndarray:
    """Solve vectorized cubic equations using MRST's closed-form layout.

    Parameters mirror ``mrstCubic.m``.  Either pass four coefficient vectors
    ``a, b, c, d`` for ``a*x**3 + b*x**2 + c*x + d = 0`` or one ``(n, 4)``
    coefficient matrix.  The result has shape ``(n, 3)`` and may contain
    complex roots; tiny imaginary roundoff is stripped just like MRST.
    """
    if b is None and c is None and d is None:
        coeff = np.asarray(a)
        if coeff.ndim != 2 or coeff.shape[1] != 4:
            raise ValueError("single-argument mrstCubic input must have shape (n, 4)")
        a, b, c, d = coeff[:, 0], coeff[:, 1], coeff[:, 2], coeff[:, 3]
    elif b is None or c is None or d is None:
        raise TypeError("mrstCubic expects either one coefficient matrix or four coefficient vectors")

    a, b, c, d = np.broadcast_arrays(_as_vector(a), _as_vector(b), _as_vector(c), _as_vector(d))
    out = np.empty((a.size, 3), dtype=complex)

    noncubic = np.isclose(a, 0.0)
    if np.any(noncubic):
        out[noncubic, :] = np.nan + 0j
        for row, coeff in zip(np.flatnonzero(noncubic), np.column_stack([b[noncubic], c[noncubic], d[noncubic]])):
            roots = np.roots(np.trim_zeros(coeff, trim="f")) if np.any(coeff) else np.array([], dtype=complex)
            roots = np.asarray(roots, dtype=complex)
            out[row, :roots.size] = roots[:3]

    cubic = ~noncubic
    if np.any(cubic):
        aa = a[cubic].astype(complex)
        bb = b[cubic].astype(complex)
        cc = c[cubic].astype(complex)
        dd = d[cubic].astype(complex)
        d0 = bb**2 - 3.0 * aa * cc
        d1 = 2.0 * bb**3 - 9.0 * aa * bb * cc + 27.0 * dd * aa**2
        disc = d1**2 - 4.0 * d0**3
        C = _complex_cuberoot(0.5 * (d1 + np.sqrt(disc)))
        bad = np.isclose(C, 0.0)
        if np.any(bad):
            C[bad] = _complex_cuberoot(0.5 * (d1[bad] - np.sqrt(disc[bad])))
        C1 = C.copy()
        C1[np.isclose(dd, 0.0) & np.isclose(d1, 0.0)] = 1.0

        u = np.array([1.0, (-1.0 + 1j * np.sqrt(3.0)) / 2.0, (-1.0 - 1j * np.sqrt(3.0)) / 2.0])
        roots = np.column_stack([-(bb + ui * C + d0 / (ui * C1)) / (3.0 * aa) for ui in u])
        out[cubic, :] = roots

    out[np.abs(out.imag) < 1e-12] = out[np.abs(out.imag) < 1e-12].real
    return out


def mrst_cubic(a, b=None, c=None, d=None) -> np.ndarray:
    """Pythonic alias for :func:`mrstCubic`."""
    return mrstCubic(a, b=b, c=c, d=d)


__all__ = ["mrstCubic", "mrst_cubic"]
