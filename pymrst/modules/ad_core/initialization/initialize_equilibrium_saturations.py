"""MRST ``ad-core/initialization/initializeEquilibriumSaturations.m``.

Saturations from equilibrium phase pressures: each non-reference phase's
saturation inverts its capillary curve at its pressure difference to the
reference phase (scaled by `pc_scale`), clipped to `s_max` above the
curve's range and set to `s_min` below it; the reference phase takes the
rest. Where that leaves it negative, the cells are set from the first and
last phases' combined capillary curve instead (the reference phase must be
neither). The inversion samples the curve on MATLAB's `0:0.0001:1` and
interpolates linearly (nearest outside), as MRST does.

`initialize_equilibrium_saturations_ad` (in `equilibrium.py`) is PyMRST's
earlier version.
"""

from __future__ import annotations

from typing import Any, Callable

import numpy as np

from pymrst.utils.matlab_interp1 import matlab_interp1

from pymrst.utils.matlab_colon import matlab_colon


def initialize_equilibrium_saturations(model: Any, region: Any,
                                       pressures: np.ndarray) -> np.ndarray:
    """MRST `s = initializeEquilibriumSaturations(model, region, pressures)`."""
    pressures = np.atleast_2d(np.asarray(pressures, dtype=float))
    nph = pressures.shape[1]
    ref = region.reference_index
    s = np.zeros(pressures.shape)
    p_ref = pressures[:, ref]
    pc_scale = _columns(region.pc_scale, pressures.shape[0], nph)
    s_min = np.atleast_2d(region.s_min)
    s_max = np.atleast_2d(region.s_max)
    for i in range(nph):
        if i == ref:
            continue
        s[:, i] = _solve_saturations(pressures[:, i], p_ref, region.pc_functions[i],
                                     region.pc_sign[i], pc_scale[:, i], s_min[:, i],
                                     s_max[:, i])
    s[:, ref] = 1 - s.sum(axis=1)
    bad = s[:, ref] < 0
    if np.any(bad):
        s[bad, :] = 0
        first, last = 0, nph - 1
        assert ref != first and ref != last
        pc = lambda S: (region.pc_sign[first]*np.asarray(region.pc_functions[first](S))
                        - region.pc_sign[last]*np.asarray(region.pc_functions[last](S)))
        s[bad, last] = _solve_saturations(pressures[bad, last], pressures[bad, first], pc, 1,
                                          pc_scale[bad, 0], s_min[:, last], s_max[:, last])
        s[bad, first] = 1 - s[bad, last]
    return s


def _columns(x: Any, nc: int, nph: int) -> np.ndarray:
    x = np.atleast_2d(np.asarray(x, dtype=float))
    return np.repeat(x, nc, axis=0) if x.shape[0] == 1 else x


def _solve_saturations(p: np.ndarray, p_ref: np.ndarray, pc_fn: Callable, pc_sign: float,
                       pc_scale: np.ndarray, s_min: np.ndarray, s_max: np.ndarray) -> np.ndarray:
    s = np.zeros(p.shape)
    sat = matlab_colon(0, 0.01, 1)
    pc = pc_sign*np.asarray(pc_fn(sat), dtype=float)
    dp = (p - p_ref)/pc_scale
    to_max = dp > np.max(pc)
    to_min = dp <= np.min(pc)
    s[to_min] = s_min[0] if s_min.size == 1 else s_min[to_min]
    s[to_max] = s_max[0] if s_max.size == 1 else s_max[to_max]
    middle = ~(to_min | to_max)
    if np.any(middle):
        s_inv = _invert_capillary(dp[middle], pc_fn, pc_sign)
        fm = np.flatnonzero(middle)
        bad = np.isnan(s_inv)
        if np.any(bad):
            s_inv[bad] = s_max[0] if s_max.size == 1 else s_max[fm[bad]]
        lo = np.full(s_inv.shape, s_min[0]) if s_min.size == 1 else s_min[middle]
        s[middle] = np.maximum(s_inv, lo)
    return s


def _invert_capillary(dp: np.ndarray, pc_fn: Callable, sgn: float) -> np.ndarray:
    s = matlab_colon(0, 0.0001, 1)
    pc = sgn*np.asarray(pc_fn(s), dtype=float)
    if pc[0] == pc[-1]:
        return np.full(dp.shape, np.nan)
    # MATLAB `[pc, ix] = unique(pc, 'last')`: sorted values, last occurrence.
    rev = pc[::-1]
    values, first_in_rev = np.unique(rev, return_index=True)
    ix = pc.size - 1 - first_in_rev
    s = s[ix]
    # `griddedInterpolant(pc, s, 'linear', 'nearest')`
    out = matlab_interp1(values, s, dp, "nearest")
    out[dp < values[0]] = s[0]
    out[dp > values[-1]] = s[-1]
    return out


initializeEquilibriumSaturations = initialize_equilibrium_saturations

__all__ = ["initializeEquilibriumSaturations", "initialize_equilibrium_saturations"]


if __name__ == "__main__":
    from types import SimpleNamespace

    def test_a_sharp_contact_without_capillarity():
        """No capillary pressure: water fills where its pressure exceeds
        oil's, oil the rest, water held at s_min above the contact."""
        region = SimpleNamespace(reference_index=1, pc_scale=np.ones((1, 2)),
                                 s_min=np.array([[0.2, 0.0]]), s_max=np.array([[1.0, 1.0]]),
                                 pc_functions=[lambda S: 0*S, lambda S: 0*S],
                                 pc_sign=np.array([-1.0, 1.0]))
        p = np.array([[9e6, 1e7], [1.1e7, 1e7]])
        s = initialize_equilibrium_saturations(None, region, p)
        np.testing.assert_allclose(s, [[0.2, 0.8], [1.0, 0.0]])
        print("ok test_a_sharp_contact_without_capillarity")

    def test_a_transition_zone_inverts_the_curve():
        """pcOW = 1e5*(1 - S): a pressure difference of -5e4 (water below
        oil) gives S = 0.5, through sign -1."""
        region = SimpleNamespace(reference_index=1, pc_scale=np.ones((1, 2)),
                                 s_min=np.array([[0.0, 0.0]]), s_max=np.array([[1.0, 1.0]]),
                                 pc_functions=[lambda S: 1e5*(1 - S), lambda S: 0*S],
                                 pc_sign=np.array([-1.0, 1.0]))
        s = initialize_equilibrium_saturations(None, region, np.array([[1e7 - 5e4, 1e7]]))
        np.testing.assert_allclose(s[0], [0.5, 0.5], atol=1e-4)
        print("ok test_a_transition_zone_inverts_the_curve")

    test_a_sharp_contact_without_capillarity()
    test_a_transition_zone_inverts_the_curve()
    print("\nAll tests passed!")
