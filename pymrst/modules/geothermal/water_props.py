"""Water and brine properties for geothermal flow (MRST geothermal/eos).

The Spivey, McCain & North (2004) correlations for the density of pure
water and NaCl brine, and the Kestin correlation for viscosity. These are
what make a geothermal simulation different from an isothermal one: at
200 C water is half as dense and a sixth as viscous as at 20 C, so the
buoyancy and the mobility both swing by more than the rock ever does.

Everything is AD-transparent - pass ADI pressures or temperatures and the
derivatives come out - because the model needs dRho/dT and dMu/dT.

Pressure in Pa, temperature in K, salinity as an NaCl **mass fraction**.
"""

from typing import Union

import numpy as np

from pymrst.utils.adi import ADI, ad_exp, ad_log, value

# validity range of the correlations (MRST's trimToEOSRange)
P_MIN, P_MAX = 0.0, 200e6                  # Pa
T_MIN_C, T_MAX_C = 0.0, 275.0              # deg C
T_MIN_C_VISC = 1e-3                        # the viscosity has a log(t)
MOL_MIN, MOL_MAX = 0.0, 5.7                # molality, mol/kg H2O
P_REF = 70e6                               # Pa, the correlation's reference

MW_NACL = 0.0584428                        # kg/mol

Num = Union[float, np.ndarray, ADI]


def _clamp(x: Num, lo: float, hi: float) -> Num:
    """Clamp into the correlation's range (MRST's trimToEOSRange).

    On an ADI this must clamp the value *and* kill the derivative outside
    the range, which is what np.clip on the value cannot do - so it is done
    by hand.
    """
    if isinstance(x, ADI):
        v = x.val
        inside = (v >= lo) & (v <= hi)
        if inside.all():
            return x
        clamped = np.clip(v, lo, hi)
        # d(clamped)/dx = 1 inside the range, 0 outside
        mask = inside.astype(np.float64)
        return ADI(clamped, [j.multiply(mask[:, None]) for j in x.jac])
    return np.clip(x, lo, hi)


def _ratio(coeff, T):
    """The (a T^2 + b T + c) / (d T^2 + e T + 1) form every coefficient takes."""
    a, b, c, d, e = coeff
    return (a * T**2 + b * T + c) / (d * T**2 + e * T + 1.0)


def _coefficients_pure_water(t_c: Num):
    """rho_w0, Ew, Fw at 70 MPa (MRST coefficients_pure_water)."""
    T = t_c / 100.0
    rho_w0 = _ratio([-0.127213, 0.645486, 1.03265, -0.070291, 0.639589], T)
    ew = _ratio([4.221, -3.478, 6.221, 0.5182, -0.4405], T)
    fw = _ratio([-11.403, 29.932, 27.952, 0.20684, 0.3768], T)
    return rho_w0, ew, fw


def _isothermal_compressibility(p: Num, p0: float, e: Num, f: Num) -> Num:
    """I = 1/E * log(E p/p0 + F), floored as MRST floors it."""
    a = e * (p / p0) + f
    tol = 1e-16
    if isinstance(a, ADI):
        low = a.val < tol
        if low.any():
            v = np.where(low, tol, a.val)
            mask = (~low).astype(np.float64)
            a = ADI(v, [j.multiply(mask[:, None]) for j in a.jac])
    else:
        a = np.where(a < tol, tol, a)
    return (1.0 / e) * ad_log(a)


def _molality(c: Num) -> Num:
    """NaCl mass fraction -> molality (mol per kg of water)."""
    x_w = 1.0 - c
    return _clamp((c / x_w) / MW_NACL, MOL_MIN, MOL_MAX)


def _coefficients_brine(ew, fw, mol, t_c):
    """Eb, Fb (MRST coefficients_brine)."""
    T = t_c / 100.0
    ecm = _ratio([0.0, 0.0, 0.1353, 0.0, 0.0], T)
    fcm32 = _ratio([-1.409, -0.361, -0.2532, 0.0, 9.216], T)
    fcm1 = _ratio([0.0, 5.614, 4.6782, -0.307, 2.6069], T)
    fcm12 = _ratio([-0.1127, 0.2047, -0.0452, 0.0, 0.0], T)

    eb = ew + ecm * mol
    fb = fw + fcm32 * mol**1.5 + fcm1 * mol + fcm12 * mol**0.5
    return eb, fb


def _density_brine_ref(rho_w0, t_c, mol):
    """Brine density at the 70 MPa reference (MRST density_brine_refP)."""
    T = t_c / 100.0
    dcm2 = _ratio([-7.925e-5, -1.93e-6, -3.42548e-4, 0.0, 0.0], T)
    dcm32 = _ratio([1.0998e-3, -2.8755e-3, -3.5819e-3, -0.72877, 1.92016], T)
    dcm1 = _ratio([-7.6402e-3, 3.6963e-2, 4.36083e-2, -0.333661, 1.185685], T)
    # MRST writes this leading coefficient as `3.746*.1e-4`, which in MATLAB
    # is 3.746e-5, not the 3.746e-4 of the paper. Kept as MRST has it, or the
    # densities would not match; it is almost certainly a typo upstream.
    dcm12 = _ratio([3.746e-5, -3.328e-4, -3.346e-4, 0.0, 0.0], T)

    return (rho_w0 + dcm2 * mol**2 + dcm32 * mol**1.5
            + dcm1 * mol + dcm12 * mol**0.5)


def density_brine(p: Num, t: Num, c: Num = 0.0) -> Num:
    """Brine density in kg/m3 (Spivey et al. 2004, eq. 15).

    p in Pa, t in K, c the NaCl mass fraction (0 = pure water).
    """
    p = _clamp(p, P_MIN, P_MAX)
    t_c = _clamp(t - 273.15, T_MIN_C, T_MAX_C)
    mol = _molality(c)

    rho_w0, ew, fw = _coefficients_pure_water(t_c)
    rho_b0 = _density_brine_ref(rho_w0, t_c, mol)
    eb, fb = _coefficients_brine(ew, fw, mol, t_c)

    ib = _isothermal_compressibility(p, P_REF, eb, fb)
    ib0 = _isothermal_compressibility(P_REF, P_REF, eb, fb)
    return rho_b0 * ad_exp(ib - ib0) * 1000.0


def density_pure_water(p: Num, t: Num) -> Num:
    """Pure-water density in kg/m3 (Spivey et al. 2004, eq. 8)."""
    p = _clamp(p, P_MIN, P_MAX)
    t_c = _clamp(t - 273.15, 1e-3, T_MAX_C)

    rho_w0, ew, fw = _coefficients_pure_water(t_c)
    iw = _isothermal_compressibility(p, P_REF, ew, fw)
    iw0 = _isothermal_compressibility(P_REF, P_REF, ew, fw)
    return rho_w0 * ad_exp(iw - iw0) * 1000.0


def _pow10(x: Num) -> Num:
    """10**x, via exp(x ln 10) so it differentiates."""
    return ad_exp(x * np.log(10.0))


def _kestin(p_mpa: Num, t_c: Num, mol: Num) -> Num:
    """Kestin et al. brine viscosity in micro-Pa s (MRST kestin_brine_viscosity)."""
    cs = 6.044 + 0.28e-2 * t_c + 0.36e-4 * t_c**2
    c_star = mol / cs
    beta_w = (-1.297 + 0.574e-1 * t_c - 0.697e-3 * t_c**2
              + 0.447e-5 * t_c**3 - 0.105e-7 * t_c**4)
    beta_se = 0.545 + 0.28e-2 * t_c - beta_w
    beta_star = 2.5 * c_star - 2.0 * c_star**2 + 0.5 * c_star**3
    beta = beta_se * beta_star + beta_w

    a = 0.3324e-1 * mol + 0.3624e-2 * mol**2 - 0.1879e-3 * mol**3
    b = -0.396e-1 * mol + 0.102e-1 * mol**2 - 0.702e-3 * mol**3

    mu_w0_20 = 1002.0                       # micro-Pa s at 20 C
    d = 20.0 - t_c
    log_ratio = (d / (96.0 + t_c)) * (1.2378 - 1.303e-3 * d
                                      + 3.06e-6 * d**2 + 2.55e-8 * d**3)
    # 10**x, written so it works on an ADI as well as an array
    mu_w0 = _pow10(log_ratio) * mu_w0_20
    mu_0 = _pow10(a + b * log_ratio) * mu_w0
    return mu_0 * (1.0 + beta * p_mpa * 1e-3)


def viscosity_brine(p: Num, t: Num, c: Num = 0.0) -> Num:
    """Brine viscosity in Pa s (Kestin et al., as MRST implements it).

    Below 125 C the Kestin correlation is used directly; above it, a
    pressure- and temperature-dependent correction is applied to the value
    at 125 C.
    """
    p = _clamp(p, P_MIN, P_MAX)
    p_mpa = p / 1e6
    t_c = _clamp(t - 273.15, T_MIN_C_VISC, T_MAX_C)
    mol = _molality(c)

    mu_125 = _kestin(p_mpa, 125.0, mol)
    mu_k = _kestin(p_mpa, t_c, mol)

    a1, a2, b1, b2 = 0.0173, 0.068, -1.0531, 0.0273
    log_t = ad_log(t_c / 125.0)
    mu_hot = mu_125 * ad_exp((a2 * (p_mpa / 100.0) + a1) * log_t**2
                             + (b2 * (p_mpa / 100.0) + b1) * log_t)

    use_kestin = value(t_c) < 125.0
    k = use_kestin.astype(np.float64)
    return (mu_hot * (1.0 - k) + mu_k * k) * 1e-6


def viscosity_pure_water(p: Num, t: Num) -> Num:
    """Pure-water viscosity in Pa s."""
    return viscosity_brine(p, t, 0.0)


if __name__ == "__main__":
    from pymrst.utils.adi import init_variables_adi
    from pymrst.utils.units import BARSA

    C = 273.15

    def test_density_at_room_conditions():
        """Water at 1 bar, 20 C is about 998 kg/m3."""
        rho = density_pure_water(np.array([1 * BARSA]), np.array([20 + C]))
        assert abs(rho[0] - 998.2) < 1.5, rho[0]
        print(f"ok test_density_at_room_conditions (rho = {rho[0]:.1f} kg/m3)")

    def test_density_falls_with_temperature_rises_with_pressure():
        """The two effects a geothermal model lives on."""
        p = np.full(4, 100 * BARSA)
        t = np.array([20.0, 100.0, 200.0, 270.0]) + C
        rho = density_pure_water(p, t)
        assert np.all(np.diff(rho) < 0), "density must fall with temperature"
        # water at 270 C is far lighter than at 20 C - this is what drives
        # the convection a geothermal doublet depends on
        assert rho[-1] / rho[0] < 0.85, rho

        hot = density_pure_water(np.array([300 * BARSA]), np.array([200 + C]))
        cold = density_pure_water(np.array([50 * BARSA]), np.array([200 + C]))
        assert hot[0] > cold[0], "density must rise with pressure"
        print(f"ok test_density_falls_with_temperature_rises_with_pressure "
              f"({rho[0]:.0f} -> {rho[-1]:.0f} kg/m3 from 20 to 270 C)")

    def test_viscosity_at_room_conditions_and_its_collapse():
        """Water at 20 C is ~1 cP, and at 250 C about a sixth of that."""
        mu20 = viscosity_pure_water(np.array([1 * BARSA]), np.array([20 + C]))
        assert abs(mu20[0] - 1.002e-3) < 5e-5, mu20[0]

        t = np.array([20.0, 100.0, 200.0, 250.0]) + C
        mu = viscosity_pure_water(np.full(4, 100 * BARSA), t)
        assert np.all(np.diff(mu) < 0), "viscosity must fall with temperature"
        assert mu[-1] / mu[0] < 0.2, mu
        print(f"ok test_viscosity_at_room_conditions_and_its_collapse "
              f"({mu[0] * 1e3:.3f} -> {mu[-1] * 1e3:.3f} cP from 20 to 250 C)")

    def test_viscosity_is_continuous_across_the_125C_switch():
        """The correlation switches formula at 125 C; it must not jump there."""
        p = np.full(2, 100 * BARSA)
        t = np.array([125.0 - 1e-6, 125.0 + 1e-6]) + C
        mu = viscosity_pure_water(p, t)
        assert abs(mu[1] - mu[0]) / mu[0] < 1e-6, (
            f"viscosity jumps at the 125 C switch: {mu}")
        print("ok test_viscosity_is_continuous_across_the_125C_switch")

    def test_salinity_raises_density_and_viscosity():
        """Brine is heavier and more viscous than water, monotonically so."""
        p = np.full(4, 100 * BARSA)
        t = np.full(4, 80 + C)
        c = np.array([0.0, 0.05, 0.15, 0.25])       # mass fraction NaCl

        rho = density_brine(p, t, c)
        mu = viscosity_brine(p, t, c)
        assert np.all(np.diff(rho) > 0), "salt must make brine heavier"
        assert np.all(np.diff(mu) > 0), "salt must make brine more viscous"
        # a saturated brine is roughly 20% denser than water
        assert 1.1 < rho[-1] / rho[0] < 1.3, rho[-1] / rho[0]

        # zero salinity must reproduce pure water exactly
        assert np.isclose(rho[0], density_pure_water(p, t)[0], rtol=1e-12)
        assert np.isclose(mu[0], viscosity_pure_water(p, t)[0], rtol=1e-12)
        print(f"ok test_salinity_raises_density_and_viscosity "
              f"(rho {rho[0]:.0f} -> {rho[-1]:.0f} kg/m3 at 25% NaCl)")

    def test_derivatives_against_finite_differences():
        """Density and viscosity differentiate through the AD layer."""
        p0 = np.array([100 * BARSA, 250 * BARSA, 50 * BARSA])
        t0 = np.array([50.0, 150.0, 240.0]) + C

        p, t = init_variables_adi(p0, t0)
        for fn in (density_pure_water, viscosity_pure_water):
            y = fn(p, t)
            for k, (x0, h) in enumerate(((p0, 1e2), (t0, 1e-4))):
                up = [p0.copy(), t0.copy()]
                dn = [p0.copy(), t0.copy()]
                up[k] += h
                dn[k] -= h
                fd = (fn(*up) - fn(*dn)) / (2 * h)
                ad = np.diag(np.asarray(y.jac[k].todense()))
                scale = max(np.abs(fd).max(), 1e-30)
                assert np.abs(ad - fd).max() / scale < 1e-5, (
                    f"{fn.__name__}, variable {k}")
        print("ok test_derivatives_against_finite_differences")

    def test_out_of_range_is_clamped_flat():
        """Outside the correlation's range the value clamps and stops moving."""
        # 400 C is well past the 275 C limit
        hot = density_pure_water(np.array([100 * BARSA]), np.array([400 + C]))
        edge = density_pure_water(np.array([100 * BARSA]), np.array([275 + C]))
        assert np.isclose(hot[0], edge[0], rtol=1e-12)

        # and the derivative there must be zero, not the unclamped slope
        p, t = init_variables_adi(np.array([100 * BARSA]), np.array([400 + C]))
        y = density_pure_water(p, t)
        assert np.isclose(np.asarray(y.jac[1].todense())[0, 0], 0.0), \
            "a clamped input must not carry a derivative"
        print("ok test_out_of_range_is_clamped_flat")

    test_density_at_room_conditions()
    test_density_falls_with_temperature_rises_with_pressure()
    test_viscosity_at_room_conditions_and_its_collapse()
    test_viscosity_is_continuous_across_the_125C_switch()
    test_salinity_raises_density_and_viscosity()
    test_derivatives_against_finite_differences()
    test_out_of_range_is_clamped_flat()
    print("\nAll tests passed!")
