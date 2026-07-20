"""Phase 329: MRST EOS Rachford-Rice and stability method surface."""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pymrst.modules.compositional import EquationOfStateModel  # noqa: E402
from pymrst.modules.compositional.eos import Mixture  # noqa: E402
from pymrst.modules.compositional.utils import solveRachfordRiceVLE  # noqa: E402


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


def validate_rachford_rice_and_labels() -> int:
    eos = EquationOfStateModel(None, _mixture(), "PR")
    z = np.array([[0.80, 0.15, 0.05], [0.45, 0.30, 0.25]])
    K = np.array([[2.0, 0.7, 0.2], [2.5, 0.8, 0.25]])
    L = eos.solveRachfordRice(None, K, z)
    V = solveRachfordRiceVLE(z, K)
    p = np.array([1.0e7, 1.2e7])
    labels = eos.singlePhaseLabel(p, np.array([240.0, 700.0]), z)

    failures = 0
    failures += _check("solveRachfordRice returns MRST liquid fraction convention",
                       np.allclose(L, 1.0 - V))
    failures += _check("singlePhaseLabel uses Li critical-temperature rule",
                       np.array_equal(labels, np.array([1.0, 0.0])))
    failures += _check("snake_case aliases mirror camelCase methods",
                       np.allclose(eos.solve_rachford_rice(None, K, z), L)
                       and np.array_equal(eos.single_phase_label(p, [240.0, 700.0], z), labels))
    return failures


def validate_kvalue_stability_branch() -> int:
    eos = EquationOfStateModel(None, _mixture(), "PR")
    eos.equilibrium_constant_functions = [
        lambda P, T, z: 0.5 + 0.0 * P,
        lambda P, T, z: 0.7 + 0.0 * P,
        lambda P, T, z: 0.8 + 0.0 * P,
    ]
    p = np.array([1.0e7, 1.1e7])
    T = np.array([350.0, 355.0])
    z = np.array([[0.80, 0.15, 0.05], [0.76, 0.18, 0.06]])
    stable, x, y = eos.performPhaseStabilityTest(p, T, z)

    failures = 0
    failures += _check("K-value stability branch marks single-phase liquid cells stable",
                       np.array_equal(stable, np.array([True, True])))
    failures += _check("K-value stability branch returns normalized liquid/vapor compositions",
                       x.shape == z.shape
                       and y.shape == z.shape
                       and np.allclose(x.sum(axis=1), 1.0)
                       and np.allclose(y.sum(axis=1), 1.0))
    failures += _check("empty stability input returns empty masks and composition matrices",
                       eos.perform_phase_stability_test(p[:0], T[:0], np.zeros((0, 3)))[0].size == 0
                       and eos.perform_phase_stability_test(p[:0], T[:0], np.zeros((0, 3)))[1].shape == (0, 3))
    return failures


def validate_custom_stability_function() -> int:
    eos = EquationOfStateModel(None, _mixture(), "PR")
    eos.equilibrium_constant_functions = [
        lambda P, T, z: 2.0 + 0.0 * P,
        lambda P, T, z: 0.7 + 0.0 * P,
        lambda P, T, z: 0.2 + 0.0 * P,
    ]
    eos.PropertyModel.checkStabilityFunction = (
        lambda P, T, z_cols: (np.array([False, True]), np.array([0.4, 1.0]))
    )
    z = np.array([[0.80, 0.15, 0.05], [0.76, 0.18, 0.06]])
    stable, x, y = eos.performPhaseStabilityTest(np.array([1.0e7, 1.1e7]), np.array([350.0, 355.0]), z)

    failures = 0
    failures += _check("PropertyModel.checkStabilityFunction hook is honored",
                       np.array_equal(stable, np.array([False, True]))
                       and x.shape == z.shape
                       and y.shape == z.shape)
    return failures


def main() -> int:
    print("\nPhase 329: MRST EOS Rachford-Rice and stability method surface")
    failures = 0
    failures += validate_rachford_rice_and_labels()
    failures += validate_kvalue_stability_branch()
    failures += validate_custom_stability_function()
    if failures:
        print(f"\nPhase 329 failed ({failures} checks).")
        return 1
    print("\nEOS Rachford-Rice and stability method parity is validated.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
