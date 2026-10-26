"""MRST ``ad-props/simple/coreyPhaseRelpermAD.m``.

A Corey relative permeability as a function of saturation:
`kr(s) = kwm*sat^n`, `sat = (s - sr)/(1 - sr_tot)` limited to [0, 1]. For AD
input MRST limits the value only, so the derivative is that of the unlimited
`sat`; the port does the same.

`corey_phase_relperm_ad_ad` is PyMRST's earlier version, `s**exponent`.
"""

from __future__ import annotations

import copy
from typing import Any, Callable, Optional

import numpy as np


def corey_phase_relperm_ad(n: float = 1, sr: float = 0, kwm: float = 1,
                           sr_tot: Optional[float] = None) -> Callable[..., Any]:
    """MRST `fn = coreyPhaseRelpermAD(n, sr, kwm, sr_tot)`: `fn(s, ...)`."""
    if sr_tot is None:
        sr_tot = sr
    return lambda s, *varargin: corey_relperm(s, n, sr, kwm, sr_tot)


def corey_relperm(s: Any, n: float, sr: float, kwm: float, sr_tot: float) -> Any:
    den = 1 - sr_tot
    sat = (s - sr)/den
    if hasattr(sat, "jac"):
        sat = copy.copy(sat)
        sat.val = np.maximum(np.minimum(sat.val, 1), 0)
    else:
        sat = np.maximum(np.minimum(sat, 1), 0)
    return kwm*sat**n


def corey_phase_relperm_ad_ad(s, exponent=1.0):
    """PyMRST's earlier version: `kr = s**exponent`."""
    return s ** exponent


coreyPhaseRelpermAD = corey_phase_relperm_ad

__all__ = ["coreyPhaseRelpermAD", "corey_phase_relperm_ad", "corey_phase_relperm_ad_ad"]


if __name__ == "__main__":
    from pymrst.utils.adi import init_variables_adi

    def test_values_are_limited_and_scaled():
        kr = corey_phase_relperm_ad(2, 0.2, 0.8, 0.4)
        s = np.array([0.1, 0.2, 0.5, 0.9])
        np.testing.assert_allclose(kr(s), 0.8*np.clip((s - 0.2)/0.6, 0, 1)**2)
        assert corey_phase_relperm_ad()(np.array([0.3]))[0] == 0.3
        print("ok test_values_are_limited_and_scaled")

    def test_ad_keeps_the_unlimited_derivative():
        kr = corey_phase_relperm_ad(2, 0.2, 1, 0.4)
        (s,) = init_variables_adi(np.array([0.1, 0.5, 0.95]))
        out = kr(s)
        sat = np.clip((s.val - 0.2)/0.6, 0, 1)
        np.testing.assert_allclose(out.val, sat**2)
        np.testing.assert_allclose(out.jac[0].diagonal(), 2*sat/0.6)
        assert out.jac[0].diagonal()[2] == 2/0.6
        print("ok test_ad_keeps_the_unlimited_derivative")

    test_values_are_limited_and_scaled()
    test_ad_keeps_the_unlimited_derivative()
    print("\nAll tests passed!")
