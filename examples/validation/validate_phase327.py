"""Phase 327: MRST EquationOfStateModel flow-property helper names."""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pymrst.modules.compositional import EquationOfStateModel  # noqa: E402
from pymrst.modules.compositional.eos import Mixture  # noqa: E402
from pymrst.modules.compositional.flow_props import (  # noqa: E402
    mass_fraction,
    mole_fraction,
    phase_saturations,
)


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


def validate_fraction_helpers() -> int:
    mix = _mixture()
    eos = EquationOfStateModel(None, mix, "PR")
    mole = np.array([[0.80, 0.15, 0.05], [0.70, 0.22, 0.08]])
    mass = mass_fraction(mix, mole)

    failures = 0
    failures += _check("getMassFraction matches PyMRST MRST-formula helper",
                       np.allclose(eos.getMassFraction(mole), mass))
    failures += _check("getMoleFraction is inverse of getMassFraction",
                       np.allclose(eos.getMoleFraction(mass), mole))
    mole_cols = [mole[:, i] for i in range(mole.shape[1])]
    mass_cols = eos.getMassFraction(mole_cols)
    roundtrip_cols = eos.getMoleFraction(mass_cols)
    failures += _check("cell/list-style fraction conversion is supported",
                       all(np.allclose(roundtrip_cols[i], mole_cols[i]) for i in range(len(mole_cols))))
    failures += _check("snake_case aliases mirror camelCase helpers",
                       np.allclose(eos.get_mass_fraction(mole), eos.getMassFraction(mole))
                       and np.allclose(eos.get_mole_fraction(mass), eos.getMoleFraction(mass)))
    return failures


def validate_saturation_helper() -> int:
    mix = _mixture()
    eos = EquationOfStateModel(None, mix, "PR")
    p = np.array([1.0e7, 1.2e7])
    T = np.array([350.0, 360.0])
    L = np.array([0.25, 0.70])
    x = np.array([[0.65, 0.25, 0.10], [0.70, 0.20, 0.10]])
    y = np.array([[0.90, 0.08, 0.02], [0.88, 0.10, 0.02]])
    Z_L = np.array([0.55, 0.60])
    Z_V = np.array([0.90, 0.95])

    s_l, s_v = eos.computeSaturations(p, T, None, None, x, y, L, Z_L, Z_V)
    ref_l, ref_v = phase_saturations(mix, p, T, L, x, y, Z_L, Z_V)
    direct_l = (L * Z_L) / (L * Z_L + (1.0 - L) * Z_V)

    failures = 0
    failures += _check("computeSaturations matches MRST no-volume-shift formula",
                       np.allclose(s_l, direct_l)
                       and np.allclose(s_v, 1.0 - direct_l))
    failures += _check("computeSaturations mirrors existing PyMRST helper",
                       np.allclose(s_l, ref_l)
                       and np.allclose(s_v, ref_v))
    failures += _check("snake_case alias returns the same saturation tuple",
                       all(np.allclose(a, b) for a, b in zip(
                           eos.compute_saturations(p, T, None, None, x, y, L, Z_L, Z_V),
                           (s_l, s_v),
                       )))
    return failures


def main() -> int:
    print("\nPhase 327: MRST EquationOfStateModel flow-property helper names")
    failures = 0
    failures += validate_fraction_helpers()
    failures += validate_saturation_helper()
    if failures:
        print(f"\nPhase 327 failed ({failures} checks).")
        return 1
    print("\nEOS flow-property helper parity is validated.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
