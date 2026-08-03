"""Phase stability test (MRST ``phaseStabilityTest.m`` layout)."""

from __future__ import annotations

import numpy as np

from pymrst.modules.compositional.eos import Mixture, fugacity_coefficients
from pymrst.modules.compositional.utils.ensure_minimum_fraction import ensureMinimumFraction


def _ensure_fraction(z, tol=1.0e-8):
    out = ensureMinimumFraction(z, tol)
    return out[0] if isinstance(out, tuple) else out
from pymrst.modules.compositional.utils.estimate_equilibrium_wilson import estimateEquilibriumWilson


# What this function actually needs from a mixture. Checked by name rather
# than by `isinstance`, because class identity is not stable here: running
# `eos.py` directly makes its `Mixture` the one defined in `__main__`, while
# this module imported `pymrst.modules.compositional.eos.Mixture` - two class
# objects from one source, and `isinstance` between them is False. That made
# `eos.py`'s own self-tests fail the moment `flash` started routing through
# this test, while the same code worked perfectly on import.
_MIXTURE_ATTRIBUTES = ("n", "Tcrit", "Pcrit", "acentric")


def _mixture_from_eos(eos) -> Mixture:
    mix = getattr(eos, "CompositionalMixture", eos)
    mix = getattr(mix, "mixture", mix)
    missing = [a for a in _MIXTURE_ATTRIBUTES if not hasattr(mix, a)]
    if missing:
        raise TypeError(
            f"phaseStabilityTest needs a Mixture-like object; "
            f"{type(mix).__name__} is missing {missing}")
    return mix


def _expand_inputs(mix: Mixture, z, p, T):
    p = np.atleast_1d(np.asarray(p, dtype=float)).reshape(-1)
    T = np.atleast_1d(np.asarray(T, dtype=float)).reshape(-1)
    z = np.atleast_2d(np.asarray(z, dtype=float))
    if z.shape[1] != mix.n:
        raise ValueError(f"composition input must have {mix.n} columns, got {z.shape[1]}")
    n = max(p.size, T.size, z.shape[0])
    if p.size not in (1, n) or T.size not in (1, n) or z.shape[0] not in (1, n):
        raise ValueError("p, T and z must have either one row or matching row counts")
    if p.size == 1:
        p = np.repeat(p, n)
    if T.size == 1:
        T = np.repeat(T, n)
    if z.shape[0] == 1:
        z = np.repeat(z, n, axis=0)
    if np.any((p <= 0.0) | ~np.isfinite(p)):
        raise AssertionError("Positive, finite pressures required for phase stability test.")
    return z, p, T


def _fugacity(mix: Mixture, p: np.ndarray, T: np.ndarray, comp: np.ndarray, liquid: bool) -> np.ndarray:
    ln_phi, _ = fugacity_coefficients(mix, p, T, comp, vapour=not liquid)
    return np.exp(ln_phi) * p[:, None] * comp


def _check_stability(
    mix: Mixture,
    z: np.ndarray,
    K: np.ndarray,
    p: np.ndarray,
    T: np.ndarray,
    inside_phase_is_vapor: bool,
    active: np.ndarray,
    tol_equil: float,
    tol_trivial: float,
    max_iterations: int,
):
    ncell = p.size
    xy = K * z if inside_phase_is_vapor else z / K
    S = np.zeros(ncell)
    trivial_solution = np.zeros(ncell, dtype=bool)
    K_work = K.copy()

    if not np.any(active):
        return xy, S, trivial_solution, K_work

    f_z_all = _fugacity(mix, p, T, z, liquid=inside_phase_is_vapor)
    active_idx = np.flatnonzero(active)

    for _ in range(max_iterations):
        if active_idx.size == 0:
            break
        z_loc = z[active_idx]
        p_loc = p[active_idx]
        T_loc = T[active_idx]
        K_loc = K_work[active_idx]
        xy_loc = K_loc * z_loc if inside_phase_is_vapor else z_loc / K_loc
        S_loc = xy_loc.sum(axis=1)
        with np.errstate(divide="ignore", invalid="ignore"):
            xy_norm = xy_loc / S_loc[:, None]
        xy_norm = _ensure_fraction(xy_norm, 1e-16)

        f_xy = _fugacity(mix, p_loc, T_loc, xy_norm, liquid=not inside_phase_is_vapor)
        f_z = f_z_all[active_idx]
        with np.errstate(divide="ignore", invalid="ignore"):
            if inside_phase_is_vapor:
                R = (f_z / f_xy) / S_loc[:, None]
            else:
                R = (f_xy / f_z) * S_loc[:, None]
        R[~np.isfinite(R)] = 1.0
        K_next = K_loc * R
        R_norm = np.sum((R - 1.0) ** 2, axis=1)
        K_norm = np.sum(np.log(np.maximum(K_next, 1e-300)) ** 2, axis=1)
        trivial = K_norm < tol_trivial
        converged = R_norm < tol_equil
        done = trivial | converged

        if np.any(done):
            done_idx = active_idx[done]
            K_work[done_idx] = K_next[done]
            S[done_idx] = S_loc[done]
            trivial_solution[done_idx] = trivial[done]
            xy[done_idx] = xy_norm[done]

        keep = ~done
        K_work[active_idx[keep]] = K_next[keep]
        active_idx = active_idx[keep]

    if active_idx.size:
        z_loc = z[active_idx]
        K_loc = K_work[active_idx]
        xy_loc = K_loc * z_loc if inside_phase_is_vapor else z_loc / K_loc
        S_loc = xy_loc.sum(axis=1)
        xy_norm = xy_loc / S_loc[:, None]
        S[active_idx] = S_loc
        xy[active_idx] = _ensure_fraction(xy_norm, 1e-16)

    return xy, S, trivial_solution, K_work


def phaseStabilityTest(
    eos,
    z,
    p,
    T,
    K=None,
    tol_equil: float = 1e-10,
    alpha=None,
    tol_trivial: float = 1e-5,
    maxIterations: int = 20000,
    solve_both: bool = True,
    **kwargs,
):
    """Perform MRST's Wilson/SSI tangent-plane-style phase stability test.

    Returns ``(stable, x, y)``.  Stable rows are single phase and have ``x`` and
    ``y`` reset to the overall composition, matching MRST's utility behavior.
    """
    if kwargs:
        unknown = ", ".join(sorted(kwargs))
        raise TypeError(f"unexpected option(s): {unknown}")
    del alpha
    mix = _mixture_from_eos(eos)
    z, p, T = _expand_inputs(mix, z, p, T)
    mc = getattr(eos, "minimumComposition", 1e-8)
    z = _ensure_fraction(z, mc)
    if K is None:
        K = estimateEquilibriumWilson(eos, p, T)
    else:
        K = np.atleast_2d(np.asarray(K, dtype=float))
        if K.shape[0] == 1 and p.size > 1:
            K = np.repeat(K, p.size, axis=0)
        if K.shape != z.shape:
            raise ValueError(f"K must have shape {z.shape}, got {K.shape}")

    active = np.ones(p.size, dtype=bool)
    y, S_v, trivial_v, _ = _check_stability(
        mix, z, K, p, T, True, active, tol_equil, tol_trivial, int(maxIterations)
    )
    vapor_stable = (S_v <= 1.0 + tol_trivial) | trivial_v

    liquid_active = np.ones(p.size, dtype=bool) if solve_both else vapor_stable.copy()
    x, S_l, trivial_l, _ = _check_stability(
        mix, z, K, p, T, False, liquid_active, tol_equil, tol_trivial, int(maxIterations)
    )
    liquid_stable = (S_l <= 1.0 + tol_trivial) | trivial_l

    stable = liquid_stable & vapor_stable
    x[stable] = z[stable]
    y[stable] = z[stable]
    bad_l = ~np.isfinite(S_l)
    bad_v = ~np.isfinite(S_v)
    x[bad_l] = z[bad_l]
    y[bad_v] = z[bad_v]
    return stable, x, y


def phase_stability_test(eos, z, p, T, K=None, **kwargs):
    """Pythonic alias for :func:`phaseStabilityTest`."""
    return phaseStabilityTest(eos, z, p, T, K=K, **kwargs)


__all__ = ["phaseStabilityTest", "phase_stability_test"]
