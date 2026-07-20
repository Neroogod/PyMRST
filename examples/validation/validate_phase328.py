"""Phase 328: base EquationOfStateModel equilibrium-constant hooks."""

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
        names=("C1", "C3", "C10"),
        Tcrit=np.array([190.56, 369.83, 617.7]),
        Pcrit=np.array([4.5992e6, 4.248e6, 2.11e6]),
        Vcrit=np.array([9.86e-5, 2.00e-4, 6.17e-4]),
        acentric=np.array([0.0114, 0.1523, 0.4884]),
        molarMass=np.array([0.01604, 0.04410, 0.14229]),
    )


def validate_matrix_input() -> int:
    eos = EquationOfStateModel(None, _mixture(), "PR")
    eos.equilibrium_constant_functions = [
        lambda P, T, z: 2.0 + 0.0 * P,
        lambda P, T, z: 0.5 + 0.0 * T,
        lambda P, T, z: np.array([1.0e12, 1.0e-12]),
    ]
    p = np.array([1.0e7, 1.2e7])
    T = 350.0
    z = np.array([[0.80, 0.15, 0.05], [0.76, 0.18, 0.06]])
    K = eos.evaluateEquilibriumConstants(p, T, z)

    failures = 0
    failures += _check("base EOS evaluates one K-function per component",
                       K.shape == z.shape
                       and np.allclose(K[:, 0], 2.0)
                       and np.allclose(K[:, 1], 0.5))
    failures += _check("K-values are clipped to MRST bounds",
                       np.allclose(K[:, 2], np.array([1.0e8, 1.0e-8])))
    failures += _check("snake_case alias matches camelCase method",
                       np.allclose(eos.evaluate_equilibrium_constants(p, T, z), K))
    return failures


def validate_cell_input_and_errors() -> int:
    eos = EquationOfStateModel(None, _mixture(), "PR")
    eos.equilibrium_constant_functions = [
        lambda P, T, z: 1.2 + 0.0 * P,
        lambda P, T, z: 0.9 + 0.0 * P,
    ]
    p = np.array([1.0e7, 1.2e7])
    T = np.array([350.0, 360.0])
    z_cols = [np.array([0.80, 0.76]), np.array([0.20, 0.24])]
    K = eos.evaluateEquilibriumConstants(p, T, z_cols)

    empty = EquationOfStateModel(None, _mixture(), "PR")
    raised = False
    try:
        empty.evaluateEquilibriumConstants(p, T, z_cols)
    except AssertionError:
        raised = True

    failures = 0
    failures += _check("cell/list input returns MRST-style list output",
                       isinstance(K, list)
                       and len(K) == 2
                       and np.allclose(K[0], 1.2)
                       and np.allclose(K[1], 0.9))
    failures += _check("missing K-functions raise the MRST-style assertion",
                       raised)
    return failures


def main() -> int:
    print("\nPhase 328: base EquationOfStateModel equilibrium-constant hooks")
    failures = 0
    failures += validate_matrix_input()
    failures += validate_cell_input_and_errors()
    if failures:
        print(f"\nPhase 328 failed ({failures} checks).")
        return 1
    print("\nBase EOS equilibrium-constant hook parity is validated.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
