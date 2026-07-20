"""MRST ``wellBoreFriction.m`` empirical wellbore friction."""

from __future__ import annotations

import numpy as np

from pymrst.utils.adi import ad_abs, ad_log, value


def _set_values_like(x, mask, replacement):
    repl = np.asarray(replacement, dtype=float)
    if repl.shape == np.asarray(mask).shape:
        repl = repl[mask]
    if hasattr(x, "val"):
        out = x.copy()
        out.val[mask] = repl
        return out
    out = np.asarray(x, dtype=float).copy()
    out[mask] = repl
    return out


def well_bore_friction(v, rho, mu, diameter, length, roughness,
                       flowtype: str = "massRate",
                       assume_turbulent: bool = False):
    """Pressure drop from MRST's empirical well-bore friction model.

    Parameters follow MRST: ``v`` is velocity, volume rate, or mass rate
    depending on ``flowtype``. The returned sign convention is the same as
    MRST: pressure loss opposes the sign of the segment velocity.
    """
    diameter = np.asarray(diameter, dtype=float)
    if diameter.size == 2 and np.asarray(value(v)).size != 2:
        di, do = float(diameter[0]), float(diameter[1])
    else:
        di, do = 0.0, diameter
    area = np.pi * ((do / 2.0) ** 2 - (di / 2.0) ** 2)
    if flowtype == "velocity":
        vel = v
    elif flowtype == "volumeRate":
        vel = v / area
    elif flowtype == "massRate":
        vel = v / (rho * area)
    else:
        raise ValueError(f"unknown flow type: {flowtype}")

    zero = np.asarray(value(vel), dtype=float) == 0.0
    vel_eff = _set_values_like(vel, zero, np.finfo(float).eps)
    re = ad_abs(rho * vel_eff * (do - di) / mu)
    f = (-3.6 * ad_log(6.9 / re + (roughness / (3.7 * do)) ** (10.0 / 9.0)) / np.log(10.0)) ** -2
    if not assume_turbulent:
        re1, re2 = 2000.0, 4000.0
        lam = np.asarray(value(re), dtype=float) <= re1
        tur = np.asarray(value(re), dtype=float) >= re2
        inter = ~(lam | tur)
        if np.any(lam):
            f = _set_values_like(f, lam, 16.0 / np.asarray(value(re), dtype=float)[lam])
        if np.any(inter):
            f2 = (-3.6 * np.log10(6.9 / re2 + (roughness / (3.7 * do)) ** (10.0 / 9.0))) ** -2
            fv = np.asarray(value(f), dtype=float).copy()
            fv[inter] = 16.0 / re1 + ((f2[inter] - 16.0 / re1) / (re2 - re1)) * (
                np.asarray(value(re), dtype=float)[inter] - re1
            )
            f = _set_values_like(f, inter, fv[inter])

    dp = -(2.0 * np.sign(value(vel_eff)) * length / (do - di)) * (f * rho * vel_eff**2)
    return _set_values_like(dp, zero, 0.0)


__all__ = ["well_bore_friction"]


if __name__ == "__main__":
    def test_well_bore_friction_sign_and_zero():
        dp = well_bore_friction(
            np.array([0.0, 2.0, -2.0]),
            rho=np.array([1000.0, 1000.0, 1000.0]),
            mu=np.array([1e-3, 1e-3, 1e-3]),
            diameter=np.array([0.1, 0.2, 0.2]),
            length=np.array([10.0, 10.0, 10.0]),
            roughness=np.array([1e-5, 1e-5, 1e-5]),
            flowtype="massRate",
        )
        assert dp[0] == 0.0
        assert dp[1] < 0.0 and dp[2] > 0.0
        print("ok test_well_bore_friction_sign_and_zero")

    test_well_bore_friction_sign_and_zero()
