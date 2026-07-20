"""CO2-brine properties from correlations (MRST co2lab-mit/utils/props).

An alternative to `co2props`, which interpolates sampled tables: here the
same quantities are *computed*, from the published correlations, so they
extend past the tables and can be differentiated by hand.

    co2_density_rk1949     Redlich-Kwong (1949) equation of state for CO2
    co2_viscosity_f1998    Fenghour et al. (1998)
    brine_density_rc1970   Rowe & Chou (1970), NaCl brine
    brine_co2_viscosity    Islam & Carlson (2012), brine with CO2 dissolved
    co2_activity_ds2003    Duan & Sun (2003) activity coefficient of CO2
    gas_mixture_viscosity  Davidson (1993)

Conventions follow the MATLAB, which are *not* the SI ones used elsewhere in
PyMRST: **temperature in K, pressure in bar**. Densities come back in kg/m3
and viscosities in Pa s. The wrappers at the bottom take SI pressure.
"""

from typing import Sequence, Tuple, Union

import numpy as np

Num = Union[float, np.ndarray]

R_BAR = 83.1447                     # bar cm3 / (mol K)
MW_CO2 = 44.0095e-3                 # kg/mol


# --------------------------------------------------------------------------
# CO2
# --------------------------------------------------------------------------
def _rk_molar_volume(t: float, p: float) -> float:
    """Molar volume of CO2 from Redlich-Kwong, in cm3/mol.

    The cubic has up to three real roots. Where it has more than one, the
    physical branch is chosen by comparing the work of expansion against the
    reversible work along the isotherm - Maxwell's equal-area construction -
    exactly as MRST does.
    """
    a_m = 7.54e7 - 4.13e4 * t                       # bar cm6 K^0.5 / mol2
    b_m = 27.8                                      # cm3/mol

    c2 = R_BAR * t / p
    c1 = (R_BAR * t * b_m) / p - a_m / (p * np.sqrt(t)) + b_m**2
    c0 = a_m * b_m / (p * np.sqrt(t))

    roots = np.roots([1.0, -c2, -c1, -c0])
    real = roots[np.abs(roots.imag) < np.finfo(float).eps].real
    if real.size == 0:
        raise ValueError(f"Redlich-Kwong found no real root at T={t}, P={p}")
    if real.size == 1:
        return float(real[0])

    v_gas, v_liq = float(real.max()), float(real.min())
    w1 = p * (v_gas - v_liq)
    w2 = (R_BAR * t * np.log((v_gas - b_m) / (v_liq - b_m))
          + a_m / (b_m * np.sqrt(t))
          * np.log(((v_gas + b_m) * v_liq) / ((v_liq + b_m) * v_gas)))
    eps = np.finfo(float).eps
    if w2 - w1 < -eps:
        return v_liq
    return v_gas                                    # ties go to the gas root


def co2_density_rk1949(T: Num, P: Num) -> np.ndarray:
    """CO2 density in kg/m3 (Redlich-Kwong 1949). T in K, P in bar.

    Tested for 283-373 K and up to 600 bar.
    """
    t = np.atleast_1d(np.asarray(T, float))
    p = np.atleast_1d(np.asarray(P, float))
    t, p = np.broadcast_arrays(t, p)
    vm = np.array([_rk_molar_volume(float(ti), float(pi))
                   for ti, pi in zip(t.ravel(), p.ravel())])
    rho_molar = (1.0 / vm) * 1e6                    # mol/m3
    return (rho_molar * MW_CO2).reshape(t.shape)


def co2_viscosity_f1998(T: Num, rho: Num) -> np.ndarray:
    """CO2 viscosity in Pa s (Fenghour et al. 1998). T in K, rho in kg/m3."""
    t = np.atleast_1d(np.asarray(T, float))
    r = np.atleast_1d(np.asarray(rho, float))
    t, r = np.broadcast_arrays(t, r)

    e = np.array([0.235156, -0.491266, 5.211155e-2, 5.347906e-2, -1.537102e-2])
    f = np.array([5.5934e-3, 6.1757e-5, 0.0, 2.6430e-11])
    g = np.array([0.4071119e-2, 0.7198037e-4, 0.2411697e-16,
                  0.2971072e-22, -0.1627888e-22])

    t_x = t / 251.196
    lt = np.log(t_x)
    psi = np.exp(sum(e[k] * lt**k for k in range(5)))

    mu = (1.00697 * np.sqrt(t) / psi
          + g[0] * r
          + g[1] * r**2
          + g[2] * r**6 / t_x**3
          + g[3] * r**8
          + g[4] * r**8 / t_x)

    # MRST's low-temperature term is `sum(f .* rho)` with a scalar rho, which
    # in MATLAB is rho * sum(f) - a single linear term, not the polynomial in
    # rho the coefficients suggest. Reproduced as written, so the numbers
    # match; it looks like a slip upstream.
    mu = mu + np.where(t < 313.15, r * f.sum(), 0.0)
    return mu / 1e6


# --------------------------------------------------------------------------
# Brine
# --------------------------------------------------------------------------
def brine_density_rc1970(T: Num, P: Num, S: Num) -> Tuple[np.ndarray, np.ndarray]:
    """NaCl brine density in kg/m3 and compressibility in 1/kPa.

    Rowe & Chou (1970). T in K, P in bar, S the NaCl **mass fraction**.
    Tested for 293-423 K and up to 350 bar.
    """
    t = np.atleast_1d(np.asarray(T, float))
    p = np.atleast_1d(np.asarray(P, float))
    s = np.atleast_1d(np.asarray(S, float))
    t, p, s = np.broadcast_arrays(t, p, s)

    bar_to_kgfcm2 = 1.01972
    q = p * bar_to_kgfcm2
    p_kpa = p * 100.0

    a1 = (5.916365 - 0.01035794 * t + 0.9270048e-5 * t**2
          - 1127.522 / t + 100674.1 / t**2)
    a2 = (0.520491e-2 - 0.10482101e-4 * t + 0.8328532e-8 * t**2
          - 1.1702939 / t + 102.2783 / t**2)
    a3 = 0.118547e-7 - 0.6599143e-10 * t
    a4 = -2.5166 + 0.0111766 * t - 0.170522e-4 * t**2
    a5 = 2.84851 - 0.0154305 * t + 0.223982e-4 * t**2
    a6 = -0.0014814 + 0.82969e-5 * t - 0.12469e-7 * t**2
    a7 = 0.0027141 - 0.15391e-4 * t + 0.22655e-7 * t**2
    a8 = 0.62158e-6 - 0.40075e-8 * t + 0.65972e-11 * t**2

    def specific_volume(qq):
        return (a1 - qq * a2 - qq**2 * a3 + a4 * s + a5 * s**2
                - qq * a6 * s - qq * a7 * s**2 - 0.5 * qq**2 * a8 * s) * 1e-3

    rho = 1.0 / specific_volume(q)

    p_ref = 101.325                                  # kPa
    r = (p_ref * 0.01) * bar_to_kgfcm2
    rho_ref = 1.0 / specific_volume(r)
    c = (rho - rho_ref) / (rho * (p_kpa - p_ref))    # 1/kPa
    return rho, c


def brine_co2_viscosity(T: Num, P: Num, m_nacl: Num, w_co2: Num) -> np.ndarray:
    """Viscosity of brine with CO2 dissolved in it, in Pa s.

    Islam & Carlson (2012). T in K, P in bar, m_nacl the NaCl molality
    (mol/kg H2O), w_co2 the dissolved CO2 mass fraction. Dissolved CO2 makes
    brine *more* viscous, by roughly 4.65 * w_co2 in relative terms.
    """
    t = np.atleast_1d(np.asarray(T, float))
    p = np.atleast_1d(np.asarray(P, float))
    m = np.atleast_1d(np.asarray(m_nacl, float))
    w = np.atleast_1d(np.asarray(w_co2, float))
    t, p, m, w = np.broadcast_arrays(t, p, m, w)

    p_mpa = p / 10.0

    # pure water density (their fit), then pure water viscosity
    a = 1.34136579e2
    b = np.array([-4.07743800e3, 1.63192756e4, 1.37091355e3])
    c = np.array([-5.56126409e-3, -1.07149234e-2, -5.46294495e-4])
    d1 = np.array([4.45861703e-1, -4.51029739e-4])

    rho_h2o = (a + sum(b[k] * 10.0 ** (c[k] * t) for k in range(3))
               + sum(d1[k] * p_mpa ** (k + 1) for k in range(2)))     # kg/m3
    rho_h2o = rho_h2o / 1e3                                           # g/cm3

    d = np.array([0.28853170e7, -0.11072577e5, -0.90834095e1,
                  0.30925651e-1, -0.27407100e-4, -0.19283851e7,
                  0.56216046e4, 0.13827250e2, -0.47609523e-1,
                  0.35545041e-4])
    e1 = np.arange(1, 6) - 3          # -2 -1 0 1 2
    e2 = np.arange(6, 11) - 8         # -2 -1 0 1 2
    mu_h2o = np.exp(sum(d[k] * t ** float(e1[k]) for k in range(5))
                    + rho_h2o * sum(d[5 + k] * t ** float(e2[k])
                                    for k in range(5)))

    # fresh water with CO2 in it
    af = np.array([1.632609119e3, -9.46077673946e2])
    bf = np.array([-1.047187396332e4, 3.68325597e1])
    mu_r = 1.0 + (af[0] * w + af[1] * w**2) / (bf[0] + bf[1] * t)
    fresh = mu_r * mu_h2o

    # brine with CO2 in it
    aa = -0.21319213 + 0.13651589e-2 * t - 0.12191756e-5 * t**2
    bb = 0.69161945e-1 - 0.27292263e-3 * t + 0.20852448e-6 * t**2
    cc = -0.25988855e-2 + 0.77989227e-5 * t
    mu_b = np.exp(aa * m + bb * m**2 + cc * m**3) * mu_h2o
    salty = mu_b * (1.0 + 4.65 * w ** 1.0134)

    return np.where(m == 0.0, fresh, salty)


def co2_activity_ds2003(T: Num, P: Num, m_ions: Sequence[float]) -> np.ndarray:
    """Activity coefficient of CO2 in brine (Duan & Sun 2003).

    T in K, P in bar. `m_ions` are six ion molalities, in MRST's order:
    two monovalent cations (Na+, K+), two divalent cations (Ca2+, Mg2+),
    then Cl- and SO4^2-. It is what turns a CO2 solubility in pure water
    into one in brine - salting-out - so it is the piece a dissolution model
    needs.
    """
    t = np.atleast_1d(np.asarray(T, float))
    p = np.atleast_1d(np.asarray(P, float))
    t, p = np.broadcast_arrays(t, p)
    m = np.asarray(m_ions, float).ravel()
    assert m.size == 6, "expected six ion molalities (Na K Ca Mg Cl SO4)"

    def coef(c1, c2, c3, c4, c5, c6):
        return (c1 + c2 * t + c3 / t + c4 * p / t
                + c5 * p / (630.0 - t) + c6 * t * np.log(p))

    lam = coef(-0.411370585, 6.07632013e-4, 97.5347708,
               -0.0237622469, 0.0170656236, 1.41335834e-5)
    zeta = coef(3.36389723e-4, -1.98298980e-5, 0.0,
                2.12222083e-3, -5.24873303e-3, 0.0)

    cations = m[0] + m[1] + 2.0 * (m[2] + m[3])      # charge-weighted
    return np.exp(2.0 * lam * cations
                  + zeta * m[4] * m[:4].sum()
                  - 0.07 * m[5])


def gas_mixture_viscosity(x: Sequence[float], M: Sequence[float],
                          mu: Sequence[float]) -> float:
    """Viscosity of a gas mixture (Davidson 1993), in the units of `mu`.

    `x` mole fractions, `M` molar masses, `mu` pure-component viscosities.
    """
    x = np.asarray(x, float).ravel()
    m = np.asarray(M, float).ravel()
    v = np.asarray(mu, float).ravel()

    num = x * np.sqrt(m)
    y = num / num.sum()
    w = x * (m / (x * m).sum())

    with np.errstate(invalid="ignore", divide="ignore"):
        e = (2.0 * np.sqrt(np.outer(w, w))) / (w[:, None] + w[None, :])
    e = np.nan_to_num(e, nan=0.0)                    # a component may be absent

    a = 0.375                                        # Davidson's best fit
    f = np.sum(np.outer(y, y) / np.sqrt(np.outer(v, v)) * e**a)
    return float(1.0 / f)


# --------------------------------------------------------------------------
# SI wrappers, for use with the rest of PyMRST
# --------------------------------------------------------------------------
def co2_density(p_pa: Num, T: Num) -> np.ndarray:
    """CO2 density in kg/m3 from SI pressure (Pa) and temperature (K)."""
    return co2_density_rk1949(T, np.asarray(p_pa, float) / 1e5)


def co2_viscosity(p_pa: Num, T: Num) -> np.ndarray:
    """CO2 viscosity in Pa s from SI pressure (Pa) and temperature (K)."""
    return co2_viscosity_f1998(T, co2_density(p_pa, T))


if __name__ == "__main__":
    C = 273.15

    def test_co2_density_against_known_states():
        """CO2 density at conditions where the answer is known.

        At 100 bar / 40 C, CO2 is a dense supercritical fluid a few hundred
        kg/m3; at 1 bar it is a light gas near 2 kg/m3. Redlich-Kwong is a
        two-parameter cubic and is not exact, but it must land in the right
        place - and it must be monotone in both variables.
        """
        gas = co2_density_rk1949(20 + C, 1.0)
        assert 1.5 < gas[0] < 2.5, gas[0]

        dense = co2_density_rk1949(40 + C, 100.0)
        assert 200.0 < dense[0] < 800.0, dense[0]

        # denser with pressure, lighter with temperature
        p = co2_density_rk1949(40 + C, np.array([50.0, 100.0, 200.0, 400.0]))
        assert np.all(np.diff(p) > 0), p
        t = co2_density_rk1949(np.array([15.0, 40.0, 80.0, 99.0]) + C, 150.0)
        assert np.all(np.diff(t) < 0), t
        print(f"ok test_co2_density_against_known_states "
              f"({gas[0]:.2f} kg/m3 at 1 bar, {dense[0]:.0f} at 100 bar/40 C)")

    def test_root_selection_picks_a_single_branch():
        """Across the phase boundary the density jumps, and stays monotone.

        Below the critical pressure the cubic has three real roots and the
        Maxwell construction has to choose. Getting that wrong shows up as a
        density that goes *backwards* with pressure.
        """
        p = np.linspace(5.0, 200.0, 60)
        rho = co2_density_rk1949(np.full_like(p, 25 + C), p)
        assert np.all(np.diff(rho) > 0), "density fell as pressure rose"
        # and it spans gas to liquid
        assert rho[0] < 20.0 and rho[-1] > 500.0, (rho[0], rho[-1])
        print(f"ok test_root_selection_picks_a_single_branch "
              f"({rho[0]:.1f} -> {rho[-1]:.0f} kg/m3 across 5-200 bar at 25 C)")

    def test_co2_viscosity():
        """CO2 viscosity is tens of micro-Pa s, and rises with density."""
        rho = co2_density_rk1949(40 + C, np.array([50.0, 100.0, 200.0]))
        mu = co2_viscosity_f1998(np.full(3, 40 + C), rho)
        assert np.all((mu > 1e-5) & (mu < 2e-4)), mu
        assert np.all(np.diff(mu) > 0), "viscosity should rise with density"
        # dilute gas: about 15 micro-Pa s
        thin = co2_viscosity_f1998(20 + C, 2.0)
        assert 1.2e-5 < thin[0] < 1.7e-5, thin[0]
        print(f"ok test_co2_viscosity "
              f"({thin[0] * 1e6:.1f} uPa.s dilute, {mu[-1] * 1e6:.1f} at 200 bar)")

    def test_brine_density_and_compressibility():
        """Rowe-Chou: salt makes brine heavier, heat makes it lighter."""
        rho, c = brine_density_rc1970(50 + C, 100.0,
                                      np.array([0.0, 0.05, 0.15, 0.25]))
        assert np.all(np.diff(rho) > 0), "salt must make brine heavier"
        assert 980.0 < rho[0] < 1010.0, rho[0]      # ~fresh water at 50 C
        assert 1150.0 < rho[-1] < 1220.0, rho[-1]   # heavy brine

        hot = brine_density_rc1970(np.array([25.0, 60.0, 120.0]) + C,
                                   100.0, 0.1)[0]
        assert np.all(np.diff(hot) < 0), "heat must make brine lighter"

        # compressibility is small and positive
        assert np.all((c > 1e-9) & (c < 1e-5)), c
        print(f"ok test_brine_density_and_compressibility "
              f"(fresh {rho[0]:.0f}, 25% NaCl {rho[-1]:.0f} kg/m3 at 50 C)")

    def test_dissolved_co2_thickens_brine():
        """CO2 in solution raises the viscosity - the 4.65 * w_co2 term."""
        t, p, m = 60 + C, 150.0, 1.0                # ~5.5% NaCl
        dry = brine_co2_viscosity(t, p, m, 0.0)
        wet = brine_co2_viscosity(t, p, m, 0.03)
        assert wet[0] > dry[0], "dissolved CO2 must thicken the brine"
        assert np.isclose(wet[0] / dry[0], 1.0 + 4.65 * 0.03 ** 1.0134,
                          rtol=1e-12)

        # salt thickens it too
        salty = brine_co2_viscosity(t, p, np.array([0.5, 1.0, 2.0]), 0.02)
        assert np.all(np.diff(salty) > 0)
        # fresh water is the m = 0 branch, which is a *different* formula.
        # Water at 60 C is about 0.47 cP, and the brine must be thicker.
        fresh = brine_co2_viscosity(t, p, 0.0, 0.02)
        assert 4.0e-4 < fresh[0] < 5.5e-4, fresh[0]
        assert dry[0] > fresh[0], "brine must be thicker than fresh water"
        print(f"ok test_dissolved_co2_thickens_brine "
              f"({dry[0] * 1e3:.3f} -> {wet[0] * 1e3:.3f} cP with 3% CO2)")

    def test_activity_coefficient_salts_co2_out():
        """More salt makes CO2 less soluble - the activity coefficient rises."""
        def na_cl(m):
            return [m, 0.0, 0.0, 0.0, m, 0.0]       # pure NaCl brine

        g = np.array([co2_activity_ds2003(60 + C, 150.0, na_cl(m))[0]
                      for m in (0.0, 0.5, 1.0, 2.0, 4.0)])
        assert np.isclose(g[0], 1.0, rtol=1e-12), \
            "no salt means no salting-out"
        assert np.all(np.diff(g) > 0), "salt must salt CO2 out"
        assert g[-1] > 1.5, g[-1]                   # 4 molal is strongly salting

        # divalent cations count double
        mono = co2_activity_ds2003(60 + C, 150.0, [2.0, 0, 0, 0, 2.0, 0])[0]
        di = co2_activity_ds2003(60 + C, 150.0, [0, 0, 1.0, 0, 2.0, 0])[0]
        assert di > mono, "a divalent cation should salt out more per mole"
        print(f"ok test_activity_coefficient_salts_co2_out "
              f"(gamma = {g[0]:.3f} fresh -> {g[-1]:.3f} at 4 molal NaCl)")

    def test_gas_mixture_viscosity():
        """Davidson: a mixture lies between its components, and reduces."""
        m = [44.0095e-3, 18.015e-3]                 # CO2, H2O
        mu = [1.5e-5, 1.0e-5]

        pure_co2 = gas_mixture_viscosity([1.0, 0.0], m, mu)
        assert np.isclose(pure_co2, mu[0], rtol=1e-12), \
            "a pure component must give back its own viscosity"

        mix = gas_mixture_viscosity([0.7, 0.3], m, mu)
        assert min(mu) < mix < max(mu), mix
        print(f"ok test_gas_mixture_viscosity "
              f"(70/30 CO2-H2O: {mix * 1e6:.2f} uPa.s)")

    def test_si_wrappers():
        """The SI wrappers just convert Pa to bar."""
        from pymrst.utils.units import BARSA
        a = co2_density(100 * BARSA, 40 + C)
        b = co2_density_rk1949(40 + C, 100.0)
        assert np.allclose(a, b, rtol=1e-14)
        mu = co2_viscosity(100 * BARSA, 40 + C)
        assert 1e-5 < mu[0] < 1e-4
        print("ok test_si_wrappers")

    test_co2_density_against_known_states()
    test_root_selection_picks_a_single_branch()
    test_co2_viscosity()
    test_brine_density_and_compressibility()
    test_dissolved_co2_thickens_brine()
    test_activity_coefficient_salts_co2_out()
    test_gas_mixture_viscosity()
    test_si_wrappers()
    print("\nAll tests passed!")
