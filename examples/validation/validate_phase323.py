"""Phase 323: EquationOfStateModel Z-factor helper parity."""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pymrst.modules.compositional import EquationOfStateModel  # noqa: E402
from pymrst.modules.compositional.eos import Mixture  # noqa: E402


def _check(name: str, condition: bool) -> int:
    print(f"  {'ok  ' if condition else 'FAIL'} {name}")
    return 0 if condition else 1


def _mixture() -> Mixture:
    return Mixture(
        names=("C1", "C3"),
        Tcrit=np.array([190.56, 369.83]),
        Pcrit=np.array([4.5992e6, 4.248e6]),
        Vcrit=np.array([9.86e-5, 2.00e-4]),
        acentric=np.array([0.0114, 0.1523]),
        molarMass=np.array([0.01604, 0.04410]),
    )


def validate_z_factor_helpers() -> int:
    eos = EquationOfStateModel(None, _mixture(), "PR")
    A = np.array([0.20, 0.40, 0.65])
    B = np.array([0.03, 0.08, 0.12])
    roots = eos.solveCubicEOS(A, B)
    roots_above_b = roots.copy()
    roots_above_b[roots_above_b < B[:, None]] = np.nan
    liquid_expected = np.nanmin(roots_above_b, axis=1)
    vapor_expected = np.nanmax(roots_above_b, axis=1)

    liquid = eos.computeLiquidZ(A, B)
    vapor = eos.computeVaporZ(A, B)
    eos.selectGibbsMinimum = False
    liquid_no_gibbs = eos.computeCompressibilityZ(
        np.full(A.size, 1.0e7),
        np.column_stack([np.full(A.size, 0.4), np.full(A.size, 0.6)]),
        A,
        B,
        np.column_stack([0.6 * A, 0.4 * A]),
        np.column_stack([0.4 * B, 0.6 * B]),
        True,
    )
    vapor_no_gibbs = eos.computeCompressibilityZ(
        np.full(A.size, 1.0e7),
        np.column_stack([np.full(A.size, 0.4), np.full(A.size, 0.6)]),
        A,
        B,
        np.column_stack([0.6 * A, 0.4 * A]),
        np.column_stack([0.4 * B, 0.6 * B]),
        False,
    )

    eos.selectGibbsMinimum = True
    composition = np.column_stack([np.full(A.size, 0.4), np.full(A.size, 0.6)])
    si = np.column_stack([0.6 * A, 0.4 * A])
    bi = np.column_stack([0.4 * B, 0.6 * B])
    z_gibbs = eos.computeCompressibilityZ(np.full(A.size, 1.0e7), composition, A, B, si, bi, True)
    fugacity, phi = eos.computeFugacity(np.full(A.size, 1.0e7), composition, vapor, A, B, si, bi)

    negative_fails = False
    try:
        eos.checkZ(np.array([1.0, -0.1]))
    except ValueError:
        negative_fails = True
    nonfinite_fails = False
    try:
        eos.checkZ(np.array([1.0, np.nan]))
    except ValueError:
        nonfinite_fails = True

    print("\nPhase 323: EquationOfStateModel Z-factor helper parity")
    failures = 0
    failures += _check("getEOSCoefficients returns cubic m1/m2",
                       eos.getEOSCoefficients() == (eos.eosA, eos.eosB))
    failures += _check("computeLiquidZ and computeVaporZ select MRST roots above B",
                       np.allclose(liquid, liquid_expected, equal_nan=True)
                       and np.allclose(vapor, vapor_expected, equal_nan=True)
                       and np.all(liquid <= vapor))
    failures += _check("computeCompressibilityZ without Gibbs delegates to liquid/vapor selectors",
                       np.allclose(liquid_no_gibbs, liquid, equal_nan=True)
                       and np.allclose(vapor_no_gibbs, vapor, equal_nan=True))
    failures += _check("computeCompressibilityZ with Gibbs returns finite selected roots",
                       z_gibbs.shape == A.shape
                       and np.all(np.isfinite(z_gibbs))
                       and np.all(z_gibbs > B))
    failures += _check("computeFugacity returns MRST-shaped fugacity and phi arrays",
                       fugacity.shape == composition.shape
                       and phi.shape == composition.shape
                       and np.all(np.isfinite(fugacity))
                       and np.all(np.isfinite(phi)))
    failures += _check("checkZ rejects negative and non-finite roots",
                       negative_fails and nonfinite_fails)
    return failures


def main() -> int:
    failures = validate_z_factor_helpers()
    if failures:
        print(f"\nPhase 323 failed ({failures} checks).")
        return 1
    print("\nEquationOfStateModel Z-factor helpers are validated.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
