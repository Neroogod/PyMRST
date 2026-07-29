"""Recover a previously eliminated AD variable block."""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np
import scipy.sparse.linalg as spla

from pymrst.utils.adi import value


def recover_vars(eq: Any, n: int, sol: Sequence[Any]) -> np.ndarray:
    """Port MRST ``recoverVars`` using a zero-based eliminated block index."""
    jac = list(eq.jac)
    if n < 0 or n >= len(jac):
        raise IndexError("eliminated block index is out of range")
    rhs = np.asarray(value(eq.val), dtype=float).ravel()
    solve = lambda b: spla.spsolve(jac[n], np.asarray(b, dtype=float).ravel())
    x = -solve(rhs)
    remaining = [index for index in range(len(jac)) if index != n]
    if len(remaining) != len(sol):
        raise ValueError("sol must contain one entry per non-eliminated block")
    for index, known in zip(remaining, sol):
        if known is not None and np.asarray(known).size and x.size:
            x = x - solve(jac[index] @ np.asarray(known, dtype=float).ravel())
    return np.asarray(x)


recoverVars = recover_vars

__all__ = ["recover_vars", "recoverVars"]
