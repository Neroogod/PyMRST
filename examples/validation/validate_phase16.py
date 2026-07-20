"""Phase 16: geothermal water/brine EOS against MRST's geothermal/eos.

432 states spanning the correlations' full range - and deliberately past it,
so the clamping is checked too.
"""

import sys
from pathlib import Path

import numpy as np
import scipy.io

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from pymrst.modules.geothermal import (                    # noqa: E402
    density_brine,
    density_pure_water,
    viscosity_brine,
    viscosity_pure_water,
)

REF = Path(__file__).with_name("ref_p16_geothermal_eos.mat")
TOL = 1e-12


def main() -> int:
    if not REF.exists():
        print(f"{REF.name} missing (run dump_phase16_reference.m in MATLAB)")
        return 0

    ref = scipy.io.loadmat(REF, squeeze_me=True)
    p = np.asarray(ref["p"], float).ravel()
    t = np.asarray(ref["T"], float).ravel()
    c = np.asarray(ref["c"], float).ravel()

    got = {
        "pure water density (kg/m3)": (density_pure_water(p, t), ref["rho_w"]),
        "pure water viscosity (Pa s)": (viscosity_pure_water(p, t), ref["mu_w"]),
        "brine density (kg/m3)": (density_brine(p, t, c), ref["rho_b"]),
        "brine viscosity (Pa s)": (viscosity_brine(p, t, c), ref["mu_b"]),
    }

    print(f"Spivey/Kestin EOS over {p.size} states "
          f"({p.min() / 1e5:.0f}-{p.max() / 1e5:.0f} bar, "
          f"{t.min() - 273.15:.0f}-{t.max() - 273.15:.0f} C, "
          f"0-{c.max():.0%} NaCl)\n")

    ok = True
    for name, (mine, theirs) in got.items():
        theirs = np.asarray(theirs, float).ravel()
        rel = np.abs(np.asarray(mine).ravel() - theirs).max() / np.abs(theirs).max()
        good = rel <= TOL
        ok &= good
        print(f"  {'ok  ' if good else 'FAIL'} {name:28s} "
              f"max rel diff = {rel:.2e}  (tol {TOL:.0e})")

    print("\nGeothermal EOS matches MATLAB!" if ok else "\ncomparisons FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
