"""Positive cubic-root helper (MRST ``cubicPositive.m`` layout)."""

from __future__ import annotations

import numpy as np


def cubicPositive(a, b=None, c=None) -> np.ndarray:
    """Solve monic cubics ``x**3 + a*x**2 + b*x + c = 0``.

    This mirrors MRST's ``cubicPositive`` helper: three-real-root rows return
    all three real roots, while one-real-root rows store that root in the first
    column and leave the remaining columns as ``NaN``.
    """
    if b is None and c is None:
        coeff = np.asarray(a, dtype=float)
        if coeff.ndim != 2 or coeff.shape[1] != 3:
            raise ValueError("single-argument cubicPositive input must have shape (n, 3)")
        a, b, c = coeff[:, 0], coeff[:, 1], coeff[:, 2]
    elif b is None or c is None:
        raise TypeError("cubicPositive expects either one coefficient matrix or three coefficient vectors")

    a, b, c = np.broadcast_arrays(np.atleast_1d(np.asarray(a, float)),
                                  np.atleast_1d(np.asarray(b, float)),
                                  np.atleast_1d(np.asarray(c, float)))
    Q = (a**2 - 3.0 * b) / 9.0
    R = (2.0 * a**3 - 9.0 * a * b + 27.0 * c) / 54.0
    M = R**2 - Q**3
    roots = np.full((c.size, 3), np.nan, dtype=float)

    neg = M < 0.0
    if np.any(neg):
        theta = np.arccos(R[neg] / np.sqrt(Q[neg]**3))
        sqrtQ = np.sqrt(Q[neg])
        aa = a[neg]
        roots[neg, 0] = -(2.0 * sqrtQ * np.cos(theta / 3.0)) - aa / 3.0
        roots[neg, 1] = -(2.0 * sqrtQ * np.cos((theta + 2.0 * np.pi) / 3.0)) - aa / 3.0
        roots[neg, 2] = -(2.0 * sqrtQ * np.cos((theta - 2.0 * np.pi) / 3.0)) - aa / 3.0

    pos = ~neg
    if np.any(pos):
        S = -np.sign(R[pos]) * np.cbrt(np.abs(R[pos]) + np.sqrt(M[pos]))
        T = Q[pos] / S
        T[~np.isfinite(T)] = 0.0
        roots[pos, 0] = S + T - a[pos] / 3.0

    return roots


def cubic_positive(a, b=None, c=None) -> np.ndarray:
    """Pythonic alias for :func:`cubicPositive`."""
    return cubicPositive(a, b=b, c=c)


__all__ = ["cubicPositive", "cubic_positive"]
