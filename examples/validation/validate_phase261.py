"""Phase 261: MRST flash utility entry-point aliases."""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pymrst.modules.compositional import EquationOfStateModel  # noqa: E402
from pymrst.modules.compositional.eos import Mixture  # noqa: E402
from pymrst.modules.compositional.utils import (  # noqa: E402
    FlashResult,
    estimateEquilibriumWilson,
    estimate_equilibrium_wilson,
    flash,
    rachford_rice,
    solveRachfordRiceVLE,
    solve_rachford_rice_vle,
    standaloneFlash,
    standalone_flash,
    wilson_k,
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


def _same_flash(a: FlashResult, b: FlashResult) -> bool:
    return (
        np.allclose(a.V, b.V)
        and np.allclose(a.x, b.x)
        and np.allclose(a.y, b.y)
        and np.allclose(a.K, b.K)
        and np.allclose(a.Z_L, b.Z_L)
        and np.allclose(a.Z_V, b.Z_V)
        and np.array_equal(a.two_phase, b.two_phase)
        and isinstance(b.iterations, int)
    )


def main() -> int:
    mix = _mixture()
    eos = EquationOfStateModel(None, mix, "Peng-Robinson")
    p = np.array([1.1e7, 1.4e7])
    T = np.array([345.0, 365.0])
    z = np.array([[0.55, 0.25, 0.20], [0.30, 0.30, 0.40]])

    K_ref = wilson_k(mix, p, T)
    K_mrst = estimateEquilibriumWilson(eos, p, T)
    K_py = estimate_equilibrium_wilson(mix, p, T)

    V_ref = rachford_rice(z, K_ref)
    V_mrst = solveRachfordRiceVLE(z, K_ref)
    V_py = solve_rachford_rice_vle(z, K_ref)

    f_ref = flash(mix, p, T, z)
    f_mrst = standaloneFlash(p, T, z, eos)
    f_py = standalone_flash(p, T, z, mix)

    print("\nPhase 261: MRST flash utility entry-point aliases")
    failures = 0
    failures += _check("Wilson camelCase accepts EquationOfStateModel",
                       K_ref.shape == (2, 3) and np.allclose(K_mrst, K_ref))
    failures += _check("Wilson snake_case accepts bare Mixture",
                       np.allclose(K_py, K_ref))
    failures += _check("Rachford-Rice camelCase matches Python kernel",
                       V_ref.shape == (2,) and np.allclose(V_mrst, V_ref))
    failures += _check("Rachford-Rice snake_case matches camelCase",
                       np.allclose(V_py, V_mrst))
    failures += _check("standaloneFlash accepts MRST argument order",
                       _same_flash(f_ref, f_mrst))
    failures += _check("standalone_flash accepts bare Mixture",
                       _same_flash(f_ref, f_py))

    if failures:
        print(f"\nPhase 261 failed ({failures} checks).")
        return 1
    print("\nMRST flash utility names now resolve to the native PyMRST implementations.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
