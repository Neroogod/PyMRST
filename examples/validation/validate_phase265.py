"""Phase 265: phaseStabilityTest utility parity."""

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
    estimateEquilibriumWilson,
    phaseStabilityTest,
    phase_stability_test,
    standaloneFlash,
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


def main() -> int:
    mix = _mixture()
    eos = EquationOfStateModel(None, mix, "Peng-Robinson")
    p = np.array([5.0e6, 5.0e6, 1.5e7])
    T = np.array([300.0, 300.0, 400.0])
    z = np.array([
        [0.70, 0.20, 0.10],
        [0.10, 0.20, 0.70],
        [0.30, 0.30, 0.40],
    ])

    flash = standaloneFlash(p, T, z, eos)
    stable, x, y = phaseStabilityTest(eos, z, p, T, maxIterations=1000)
    K = estimateEquilibriumWilson(eos, p, T)
    stable_k, x_k, y_k = phase_stability_test(eos, z, p, T, K=K, maxIterations=1000)

    expected_stable = ~flash.two_phase
    print("\nPhase 265: phaseStabilityTest utility parity")
    failures = 0
    failures += _check("classification matches native flash two-phase flag",
                       np.array_equal(stable, expected_stable)
                       and stable.dtype == np.dtype(bool))
    failures += _check("stable rows return overall composition for both trial phases",
                       np.allclose(x[stable], z[stable])
                       and np.allclose(y[stable], z[stable]))
    failures += _check("unstable rows return distinct normalized phase estimates",
                       np.allclose(x[~stable].sum(axis=1), 1.0)
                       and np.allclose(y[~stable].sum(axis=1), 1.0)
                       and np.all(np.linalg.norm(x[~stable] - y[~stable], axis=1) > 1e-3))
    failures += _check("explicit Wilson K input matches estimated-K path",
                       np.array_equal(stable_k, stable)
                       and np.allclose(x_k, x)
                       and np.allclose(y_k, y))
    failures += _check("single-row composition broadcasts over p/T vectors",
                       phaseStabilityTest(eos, z[0], p[:2], T[:2], maxIterations=1000)[0].shape == (2,))

    if failures:
        print(f"\nPhase 265 failed ({failures} checks).")
        return 1
    print("\nphaseStabilityTest now performs the native SSI stability test on PyMRST EOS mixtures.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
