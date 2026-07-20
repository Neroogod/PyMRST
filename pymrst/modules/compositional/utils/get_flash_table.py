"""Flash table utility (MRST ``getFlashTable.m`` layout)."""

from __future__ import annotations

import numpy as np

from pymrst.modules.compositional.utils.standalone_flash import standaloneFlash


def _mixture_from_model(model):
    mix = getattr(model, "CompositionalMixture", model)
    return getattr(mix, "mixture", mix)


def _composition_matrix(z, n: int, ncomp: int) -> np.ndarray:
    if isinstance(z, (list, tuple)) and len(z) == ncomp and not np.isscalar(z[0]):
        cols = []
        for zi in z:
            arr = np.asarray(zi, dtype=float).reshape(-1)
            if arr.size == 1:
                arr = np.repeat(arr, n)
            if arr.size != n:
                raise ValueError(f"component stream entries must have length 1 or {n}, got {arr.size}")
            cols.append(arr)
        out = np.column_stack(cols)
    else:
        arr = np.asarray(z, dtype=float)
        if arr.ndim == 1:
            if arr.size != ncomp:
                raise ValueError(f"composition vector must have {ncomp} entries, got {arr.size}")
            out = np.repeat(arr.reshape(1, -1), n, axis=0)
        elif arr.ndim == 2:
            if arr.shape[1] != ncomp:
                raise ValueError(f"composition matrix must have {ncomp} columns, got {arr.shape[1]}")
            if arr.shape[0] == 1:
                out = np.repeat(arr, n, axis=0)
            elif arr.shape[0] == n:
                out = arr.copy()
            else:
                raise ValueError(f"composition matrix must have 1 or {n} rows, got {arr.shape[0]}")
        else:
            raise ValueError("composition input must be a vector, matrix, or component sequence")
    total = out.sum(axis=1, keepdims=True)
    if np.any(total <= 0.0):
        raise ValueError("composition rows must have positive total amount")
    return out / total


def _fix(vec: np.ndarray, nT: int, np_: int) -> np.ndarray:
    return np.asarray(vec).reshape((nT, np_), order="F").T


def getFlashTable(p_range, T_range, z, EOS):
    """Evaluate standalone flash over a pressure-temperature mesh.

    Returns ``(L, x, y, Z_L, Z_V, p, T)`` following MRST's utility: ``L`` is
    liquid fraction, ``x`` and ``y`` are component-table lists, and all tables
    are shaped ``(len(p_range), len(T_range))``.
    """
    p_range = np.asarray(p_range, dtype=float).reshape(-1)
    T_range = np.asarray(T_range, dtype=float).reshape(-1)
    P, TT = np.meshgrid(p_range, T_range)
    np_ = p_range.size
    nT = T_range.size
    n = P.size
    mix = _mixture_from_model(EOS)
    Z = _composition_matrix(z, n, mix.n)

    result = standaloneFlash(P.reshape(-1, order="F"), TT.reshape(-1, order="F"), Z, EOS)
    L = _fix(1.0 - result.V, nT, np_)
    Z_L = _fix(result.Z_L, nT, np_)
    Z_V = _fix(result.Z_V, nT, np_)
    x = [_fix(result.x[:, i], nT, np_) for i in range(mix.n)]
    y = [_fix(result.y[:, i], nT, np_) for i in range(mix.n)]
    p = _fix(P.reshape(-1, order="F"), nT, np_)
    T = _fix(TT.reshape(-1, order="F"), nT, np_)
    return L, x, y, Z_L, Z_V, p, T


def get_flash_table(p_range, T_range, z, EOS):
    """Pythonic alias for :func:`getFlashTable`."""
    return getFlashTable(p_range, T_range, z, EOS)


__all__ = ["getFlashTable", "get_flash_table"]
