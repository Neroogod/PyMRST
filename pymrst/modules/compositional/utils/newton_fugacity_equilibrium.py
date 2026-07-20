"""Newton flash-equilibrium update (MRST ``newtonFugacityEquilibrium.m``)."""

from __future__ import annotations

from typing import Any

import numpy as np

from pymrst.modules.compositional.eos import fugacity_coefficients
from pymrst.modules.compositional.eos_ad import equations_equilibrium, _jac_diagonals
from pymrst.modules.compositional.utils.ensure_minimum_fraction import ensureMinimumFraction
from pymrst.utils.adi import init_variables_adi, value


def _eos_model(model: Any):
    eos = getattr(model, "EOSModel", None)
    if eos is None and isinstance(model, dict):
        eos = model.get("EOSModel")
    return model if eos is None else eos


def _mixture(eos):
    mix = getattr(eos, "CompositionalMixture", None)
    if mix is None:
        mix = getattr(eos, "mixture", None)
    if mix is None:
        mix = getattr(eos, "fluid", None)
    if mix is None:
        raise ValueError("model/EOSModel must expose a compositional mixture")
    return mix


def _as_cells(value_in, ncomp: int | None = None) -> np.ndarray:
    arr = np.asarray(value_in, dtype=float)
    if arr.ndim == 0:
        arr = arr.reshape(1)
    if ncomp is not None:
        arr = np.atleast_2d(arr)
        if arr.shape[1] != ncomp and arr.shape[0] == ncomp:
            arr = arr.T
    return arr


def _normalize_rows(arr: np.ndarray, floor: float = 1.0e-300) -> np.ndarray:
    arr = np.maximum(np.asarray(arr, dtype=float), floor)
    denom = arr.sum(axis=1, keepdims=True)
    return np.divide(arr, denom, out=np.full_like(arr, 1.0 / arr.shape[1]), where=denom != 0.0)


def _compute_xy(z: np.ndarray, K: np.ndarray, L: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    denom = L[:, None] + (1.0 - L)[:, None] * K
    x = z / np.maximum(denom, 1.0e-300)
    y = K * x
    return _normalize_rows(x), _normalize_rows(y)


def _relax(values: np.ndarray, delta: np.ndarray) -> float:
    values = np.asarray(values, dtype=float).reshape(-1)
    delta = np.asarray(delta, dtype=float).reshape(-1)
    if values.size == 0:
        return 1.0
    weights = np.ones_like(values)
    nz = np.abs(delta) > 0.0
    proposed = values[nz] + delta[nz]
    clipped = np.clip(proposed, 0.0, 1.0)
    with np.errstate(divide="ignore", invalid="ignore"):
        weights[nz] = (clipped - values[nz]) / delta[nz]
    weights[~np.isfinite(weights)] = 1.0
    weights = np.clip(weights, 0.0, 1.0)
    return float(weights.min(initial=1.0))


def _solve_cell_updates(dFds: np.ndarray, residual: np.ndarray) -> np.ndarray:
    ncell = dFds.shape[0]
    out = np.zeros((ncell, dFds.shape[2]), dtype=float)
    for c in range(ncell):
        A = dFds[c]
        b = -residual[c]
        try:
            out[c] = np.linalg.solve(A, b)
        except np.linalg.LinAlgError:
            out[c] = np.linalg.lstsq(A, b, rcond=None)[0]
    return out


def _fugacity_ratio_values(mix, p, T, x, y):
    ln_phi_l, Z_L = fugacity_coefficients(mix, p, T, x, vapour=False)
    ln_phi_v, Z_V = fugacity_coefficients(mix, p, T, y, vapour=True)
    f_l = np.exp(ln_phi_l) * p[:, None] * np.maximum(x, 1.0e-300)
    f_v = np.exp(ln_phi_v) * p[:, None] * np.maximum(y, 1.0e-300)
    return f_l, f_v, Z_L, Z_V


def newtonFugacityEquilibrium(model, P, T, z, K, L):
    """Perform one Newton update for liquid/vapor flash equilibrium.

    This follows MRST's ``newtonFugacityEquilibrium`` layout: secondary
    variables are liquid mole fraction ``L`` plus liquid/vapor compositions
    ``x`` and ``y``. The equilibrium residual and Jacobian are assembled with
    PyMRST AD and solved independently per cell.
    """
    eos = _eos_model(model)
    mix = _mixture(eos)
    ncomp = int(mix.n)
    p = np.asarray(P, dtype=float).reshape(-1)
    T_arr = np.asarray(T, dtype=float).reshape(-1)
    if T_arr.size == 1 and p.size > 1:
        T_arr = np.full(p.size, float(T_arr[0]))
    z_floor, _ = ensureMinimumFraction(_as_cells(z, ncomp))
    z_arr = _normalize_rows(z_floor)
    K_arr = np.maximum(_as_cells(K, ncomp), 1.0e-300)
    if K_arr.shape[0] == 1 and p.size > 1:
        K_arr = np.repeat(K_arr, p.size, axis=0)
    if z_arr.shape[0] == 1 and p.size > 1:
        z_arr = np.repeat(z_arr, p.size, axis=0)
    L_arr = np.clip(np.asarray(L, dtype=float).reshape(-1), 0.0, 1.0)
    if L_arr.size == 1 and p.size > 1:
        L_arr = np.full(p.size, float(L_arr[0]))
    if not (p.size == T_arr.size == z_arr.shape[0] == K_arr.shape[0] == L_arr.size):
        raise ValueError("P, T, z, K and L must describe the same number of cells")

    x, y = _compute_xy(z_arr, K_arr, L_arr)
    _, _, Z_L, Z_V = _fugacity_ratio_values(mix, p, T_arr, x, y)

    sec = init_variables_adi(L_arr, *[x[:, i] for i in range(ncomp)], *[y[:, i] for i in range(ncomp)])
    L_ad = sec[0]
    x_ad = list(sec[1:1 + ncomp])
    y_ad = list(sec[1 + ncomp:1 + 2 * ncomp])
    z_cols = [z_arr[:, i] for i in range(ncomp)]
    eqs = equations_equilibrium(mix, p, T_arr, x_ad, y_ad, z_cols, L_ad, Z_L, Z_V)
    residual = np.column_stack([np.asarray(value(eq), dtype=float).reshape(-1) for eq in eqs])
    dFds = _jac_diagonals(eqs, 1 + 2 * ncomp)
    upd = _solve_cell_updates(dFds, residual)

    dL = upd[:, 0]
    dx = upd[:, 1:1 + ncomp]
    dy = upd[:, 1 + ncomp:1 + 2 * ncomp]
    dL[~np.isfinite(dL)] = np.sign(dL[~np.isfinite(dL)])

    w = _relax(L_arr, dL)
    for i in range(ncomp):
        dx[L_arr == 0.0, i] = 0.0
        dy[L_arr == 1.0, i] = 0.0
        w = min(w, _relax(x[:, i], dx[:, i]), _relax(y[:, i], dy[:, i]))

    L_new = np.clip(L_arr + w * dL, 0.0, 1.0)
    x_new = x + w * dx
    y_new = y + w * dy
    pure = (L_new >= 1.0 - 1.0e-12) | (L_new <= 1.0e-12)
    x_new[pure] = z_arr[pure]
    y_new[pure] = z_arr[pure]
    x_new = _normalize_rows(x_new)
    y_new = _normalize_rows(y_new)

    f_l, f_v, Z_L_new, Z_V_new = _fugacity_ratio_values(mix, p, T_arr, x_new, y_new)
    ratio = np.divide(f_l, f_v, out=np.ones_like(f_l), where=f_v != 0.0)
    ratio[pure[:, None] | (z_arr <= 0.0)] = 1.0
    K_new = np.divide(y_new, x_new, out=np.ones_like(y_new), where=x_new > 0.0)
    gone = (z_arr <= 0.0) | (x_new <= 0.0) | pure[:, None]
    K_new[gone] = 1.0
    values = np.abs(ratio - 1.0)
    return x_new, y_new, K_new, Z_L_new, Z_V_new, L_new, values


def newton_fugacity_equilibrium(model, P, T, z, K, L):
    """Pythonic alias for :func:`newtonFugacityEquilibrium`."""
    return newtonFugacityEquilibrium(model, P, T, z, K, L)


__all__ = ["newtonFugacityEquilibrium", "newton_fugacity_equilibrium"]
