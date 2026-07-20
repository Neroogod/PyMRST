"""Phase 324: cubic EOS getCompressibilityAndFugacity parity."""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pymrst.modules.compositional import EquationOfStateModel  # noqa: E402
from pymrst.modules.compositional.eos import Mixture, fugacity_coefficients  # noqa: E402


def _check(name: str, condition: bool) -> int:
    print(f"  {'ok  ' if condition else 'FAIL'} {name}")
    return 0 if condition else 1


def _mixture() -> Mixture:
    return Mixture(
        names=("C1", "C3", "C10"),
        Tcrit=np.array([190.56, 369.83, 617.7]),
        Pcrit=np.array([4.5992e6, 4.248e6, 2.11e6]),
        Vcrit=np.array([9.86e-5, 2.00e-4, 6.17e-4]),
        acentric=np.array([0.0114, 0.1523, 0.4884]),
        molarMass=np.array([0.01604, 0.04410, 0.14229]),
    )


def validate_compressibility_and_fugacity() -> int:
    mix = _mixture()
    eos = EquationOfStateModel(None, mix, "PR")
    eos.selectGibbsMinimum = False
    p = np.array([1.0e7, 1.2e7])
    T = np.array([350.0, 360.0])
    x = np.array([[0.70, 0.20, 0.10], [0.65, 0.25, 0.10]])
    y = np.array([[0.90, 0.08, 0.02], [0.88, 0.10, 0.02]])
    z = np.array([[0.80, 0.15, 0.05], [0.76, 0.18, 0.06]])

    Z_L, Z_V, f_L, f_V = eos.getCompressibilityAndFugacity(p, T, x, y, z)
    ln_phi_l, ref_Z_L = fugacity_coefficients(mix, p, T, x, vapour=False)
    ln_phi_v, ref_Z_V = fugacity_coefficients(mix, p, T, y, vapour=True)
    ref_f_L = np.exp(ln_phi_l) * p[:, None] * x
    ref_f_V = np.exp(ln_phi_v) * p[:, None] * y

    supplied_Z_L = np.array([0.55, 0.60])
    supplied_Z_V = np.array([0.85, 0.90])
    keep_Z_L, keep_Z_V, keep_f_L, keep_f_V = eos.get_compressibility_and_fugacity(
        p,
        T,
        x,
        y,
        z,
        supplied_Z_L,
        supplied_Z_V,
    )
    si_l, si_v, a_l, a_v, b_l, b_v, bi = eos.getMixtureFugacityCoefficients(p, T, x, y)

    print("\nPhase 324: cubic EOS getCompressibilityAndFugacity parity")
    failures = 0
    failures += _check("getMixtureFugacityCoefficients returns MRST-shaped arrays",
                       si_l.shape == x.shape
                       and si_v.shape == y.shape
                       and a_l.shape == p.shape
                       and a_v.shape == p.shape
                       and b_l.shape == p.shape
                       and b_v.shape == p.shape
                       and bi.shape == x.shape)
    failures += _check("getCompressibilityAndFugacity Z factors match established PR helper",
                       np.allclose(Z_L, ref_Z_L)
                       and np.allclose(Z_V, ref_Z_V))
    failures += _check("getCompressibilityAndFugacity fugacity values match established PR helper",
                       np.allclose(f_L, ref_f_L)
                       and np.allclose(f_V, ref_f_V))
    failures += _check("supplied Z values are preserved through setZDerivatives hook",
                       np.allclose(keep_Z_L, supplied_Z_L)
                       and np.allclose(keep_Z_V, supplied_Z_V)
                       and keep_f_L.shape == x.shape
                       and keep_f_V.shape == y.shape)
    failures += _check("snake_case alias returns same tuple shapes",
                       keep_Z_L.shape == Z_L.shape
                       and keep_Z_V.shape == Z_V.shape)
    return failures


def main() -> int:
    failures = validate_compressibility_and_fugacity()
    if failures:
        print(f"\nPhase 324 failed ({failures} checks).")
        return 1
    print("\nCubic EOS compressibility and fugacity helper is validated.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
