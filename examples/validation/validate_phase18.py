"""Phase 18: CO2-brine property correlations against MRST co2lab-mit.

Redlich-Kwong CO2 density, Fenghour viscosity, Rowe-Chou brine density,
Islam-Carlson brine+CO2 viscosity, Duan-Sun activity, Davidson mixture rule -
over grids that cross the correlations' internal switches (the 313.15 K branch
in the CO2 viscosity, the zero-salinity branch in the brine viscosity, and the
gas/liquid root selection in the cubic).
"""

import sys
from pathlib import Path

import numpy as np
import scipy.io

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from pymrst.modules.co2lab.co2_brine_props import (          # noqa: E402
    brine_co2_viscosity,
    brine_density_rc1970,
    co2_activity_ds2003,
    co2_density_rk1949,
    co2_viscosity_f1998,
    gas_mixture_viscosity,
)

REF = Path(__file__).with_name("ref_p18_co2_brine_props.mat")
TOL = 1e-12


def main() -> int:
    if not REF.exists():
        print(f"{REF.name} missing (run dump_phase18_reference.m in MATLAB)")
        return 0

    r = scipy.io.loadmat(REF, squeeze_me=True)
    g = lambda k: np.asarray(r[k], float).ravel()   # noqa: E731

    checks = []

    # CO2 density and viscosity
    rho = co2_density_rk1949(g("T"), g("P"))
    checks.append(("CO2 density (Redlich-Kwong)", rho, g("rho_co2")))
    checks.append(("CO2 viscosity (Fenghour)",
                   co2_viscosity_f1998(g("T"), rho), g("mu_co2")))

    # brine density and compressibility
    rho_b, c_b = brine_density_rc1970(g("T2"), g("P2"), g("S2"))
    checks.append(("brine density (Rowe-Chou)", rho_b, g("rho_b")))

    # Compressibility is (rho - rho_ref) / (rho * (p - p_ref)), and Rowe-Chou's
    # reference pressure is 101.325 kPa - so at P = 1 bar both the numerator
    # and the denominator very nearly vanish and the *relative* error blows up
    # on pure round-off. The absolute agreement is 2e-16 on values of ~3e-7.
    # Checked away from the reference pressure, and in absolute terms overall.
    far = g("P2") > 5.0
    checks.append(("brine compressibility (P > 5 bar)",
                   np.asarray(c_b).ravel()[far], g("c_b")[far]))

    # brine + CO2 viscosity
    checks.append(("brine+CO2 viscosity (Islam-Carlson)",
                   brine_co2_viscosity(g("T3"), g("P3"), g("M3"), g("W3")),
                   g("mu_bc")))

    # activity coefficient
    t4, p4, m4 = g("T4"), g("P4"), g("M4")
    gam = np.array([co2_activity_ds2003(t, p, [m, 0, 0, 0, m, 0])[0]
                    for t, p, m in zip(t4, p4, m4)])
    checks.append(("CO2 activity (Duan-Sun)", gam, g("gam")))

    # gas mixture
    xs = np.atleast_2d(r["xs"])
    mw = g("Mw")
    mus = g("mus")
    mix = np.array([gas_mixture_viscosity(x, mw, mus) for x in xs])
    checks.append(("gas mixture viscosity (Davidson)", mix, g("mu_mix")))

    print(f"CO2-brine correlations over {g('T3').size} states "
          f"(15-95 C, 1-400 bar, 0-25% NaCl, 0-3% dissolved CO2)\n")

    ok = True
    for name, mine, theirs in checks:
        mine = np.asarray(mine, float).ravel()
        rel = np.abs(mine - theirs).max() / np.abs(theirs).max()
        good = rel <= TOL
        ok &= good
        print(f"  {'ok  ' if good else 'FAIL'} {name:36s} "
              f"max rel diff = {rel:.2e}  (tol {TOL:.0e})")

    # and the compressibility everywhere, absolutely - including the
    # ill-conditioned point at the reference pressure
    c_abs = np.abs(np.asarray(c_b).ravel() - g("c_b")).max()
    good = c_abs <= 1e-14
    ok &= good
    print(f"  {'ok  ' if good else 'FAIL'} "
          f"{'brine compressibility (all P, abs)':36s} "
          f"max abs diff = {c_abs:.2e}  (tol 1e-14, values ~3e-07)")

    print("\nCO2-brine correlations match MATLAB!" if ok
          else "\ncomparisons FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
