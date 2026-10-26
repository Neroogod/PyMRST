"""MRST ``ad-core/models/facilities/wellBoreFriction.m``.

The frictional pressure drop along a pipe segment: from the flow `v` (a
velocity, volume rate or, by default, mass rate) through an annulus or pipe
of diameter `D` (`[Di, Do]` for an annulus), the Reynolds number gives the
Haaland friction factor `f = (-3.6 log10(6.9/Re + (e/3.7Do)^(10/9)))^-2`.
Unless `assumeTurbulent`, the laminar region (Re <= 2000) takes `16/Re` and
the intermediate one (2000 < Re < 4000) interpolates linearly between them,
as ECLIPSE does. The drop is `-(2 sign(v) L/(Do-Di)) f rho v^2`; a zero flow
is evaluated at `eps` and its drop value set to zero (its derivative kept).

`well_bore_friction_ad`, below, is PyMRST's earlier version.
"""

from __future__ import annotations

import numpy as np

from pymrst.utils.adi import ad_abs, ad_log, is_ad, value

LAMINAR_REYNOLDS: float = 2000.0
TURBULENT_REYNOLDS: float = 4000.0


def well_bore_friction(v, rho, mu, D, L, roughness, flowtype: str = "massRate",
                       assumeTurbulent: bool = False):
    """MRST `dp = wellBoreFriction(v, rho, mu, D, L, roughness, flowtype, assumeTurbulent)`."""
    D = np.asarray(D, dtype=float)
    if D.size == 2:
        Di, Do = D[0], D[1]
    else:
        Di, Do = 0.0, D
    if flowtype == "velocity":
        pass
    elif flowtype == "volumeRate":
        v = v/(np.pi*((Do/2)**2 - (Di/2)**2))
    elif flowtype == "massRate":
        v = v/(np.pi*rho*((Do/2)**2 - (Di/2)**2))
    else:
        raise ValueError(f"Unknown flow type: {flowtype}")
    is_zero = np.asarray(value(v)) == 0
    v = v.copy() if is_ad(v) else np.array(v, dtype=float)
    v[is_zero] = np.finfo(float).eps
    re = ad_abs(rho*v*(Do - Di)/mu)
    f = (-3.6*ad_log(6.9/re + (roughness/(3.7*Do))**(10/9))/np.log(10))**(-2)
    if not assumeTurbulent:
        re1, re2 = LAMINAR_REYNOLDS, TURBULENT_REYNOLDS
        re_val = np.asarray(value(re))
        lam = re_val <= re1
        tur = re_val >= re2
        inter = ~(lam | tur)
        f1 = 16/re1
        f2 = (-3.6*np.log10(6.9/re2 + (roughness/(3.7*Do))**(10/9)))**(-2)
        if np.any(lam):
            f[lam] = 16/re[lam]
        if np.any(inter):
            if np.size(f2) > np.count_nonzero(inter):
                f2 = np.asarray(f2)[inter]
            f[inter] = f1 + ((f2 - f1)/(re2 - re1))*(re[inter] - re1)
    dp = -(2*np.sign(value(v))*L/(Do - Di))*(f*rho*v**2)
    if is_ad(dp):
        dp.val[is_zero] = 0
    else:
        dp[is_zero] = 0
    return dp


wellBoreFriction = well_bore_friction


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


def well_bore_friction_ad(v, rho, mu, diameter, length, roughness,
                           flowtype: str = "massRate",
                           assume_turbulent: bool = False):
    """PyMRST's earlier well-bore friction (derivatives of the laminar and
    intermediate branches are not MRST's).

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


__all__ = ["well_bore_friction", "well_bore_friction_ad", "wellBoreFriction"]


if __name__ == "__main__":
    from pymrst.utils.adi import init_variables_adi

    def test_well_bore_friction_sign_and_zero():
        dp = well_bore_friction_ad(
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

    def test_the_three_regions_and_a_zero_flow():
        # Mass rates giving Re about 1e3, 3e3 and 1e5 in a 0.1 m pipe of water.
        area = np.pi*0.05**2
        rate = np.array([0.0, 1e3, 3e3, 1e5])*1e-3*area/0.1
        rho, mu = np.full(4, 1000.0), np.full(4, 1e-3)
        dp = well_bore_friction(rate, rho, mu, 0.1, 10.0, 1e-4)
        assert dp[0] == 0.0 and np.all(dp[1:] < 0)
        vel = rate[1]/(rho[1]*area)
        np.testing.assert_allclose(dp[1], -(2*10/0.1)*(16/1e3)*1000*vel**2, rtol=1e-12)
        turbulent = well_bore_friction(rate, rho, mu, 0.1, 10.0, 1e-4, "massRate", True)
        assert turbulent[1] != dp[1] and turbulent[3] == dp[3]
        print("ok test_the_three_regions_and_a_zero_flow")

    def test_the_derivative_follows_the_laminar_branch():
        area = np.pi*0.05**2
        q = init_variables_adi(np.array([1e3*1e-3*area/0.1]))[0]
        dp = well_bore_friction(q, 1000.0, 1e-3, 0.1, 10.0, 1e-4)
        # Laminar: dp = -c q (f = 16/Re is 1/q), so d(dp)/dq = dp/q.
        np.testing.assert_allclose(dp.jac[0].toarray()[0, 0], dp.val[0]/q.val[0], rtol=1e-10)
        print("ok test_the_derivative_follows_the_laminar_branch")

    def test_an_annulus_and_an_unknown_flow_type():
        dp = well_bore_friction(np.array([1.0]), 1000.0, 1e-3, [0.05, 0.1], 10.0, 1e-4)
        assert dp[0] < 0
        try:
            well_bore_friction(np.array([1.0]), 1000.0, 1e-3, 0.1, 10.0, 1e-4, "bad")
        except ValueError:
            print("ok test_an_annulus_and_an_unknown_flow_type")
            return
        raise AssertionError("unknown flow type must raise")

    test_well_bore_friction_sign_and_zero()
    test_the_three_regions_and_a_zero_flow()
    test_the_derivative_follows_the_laminar_branch()
    test_an_annulus_and_an_unknown_flow_type()
    print("\nAll tests passed!")
