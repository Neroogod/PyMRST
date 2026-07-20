"""Initialize FastAD variables (MRST ``initVariablesFastAD.m``)."""

from __future__ import annotations

import numpy as np

from pymrst.modules.compositional.utils.fast_ad import FastAD


def initVariablesFastAD(*args):
    """Return one FastAD variable per input with dense identity Jacobian blocks."""
    if not args:
        return ()
    vals = [np.asarray(arg, dtype=float).reshape(-1) for arg in args]
    m = vals[0].size
    if any(v.size != m for v in vals):
        raise AssertionError("All FastAD input variables must have the same number of elements")
    n = len(vals)
    out = []
    for i, val in enumerate(vals):
        jac = np.zeros((m, n), dtype=float)
        jac[:, i] = 1.0
        out.append(FastAD(val, jac))
    return tuple(out)


def init_variables_fast_ad(*args):
    """Pythonic alias for :func:`initVariablesFastAD`."""
    return initVariablesFastAD(*args)


__all__ = ["initVariablesFastAD", "init_variables_fast_ad"]
