"""Phase 255: Mixture MRST CompositionalMixture API aliases."""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

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


def main() -> int:
    mix = _mixture()

    vector_setter_returns_self = mix.setBinaryInteraction([0.01, 0.02, 0.03]) is mix
    bic_after_vector = mix.getBinaryInteraction().copy()
    expected_vector_bic = np.array(
        [
            [0.0, 0.01, 0.02],
            [0.01, 0.0, 0.03],
            [0.02, 0.03, 0.0],
        ]
    )

    matrix = np.array(
        [
            [0.0, 0.04, 0.05],
            [0.04, 0.0, 0.06],
            [0.05, 0.06, 0.0],
        ]
    )
    matrix_setter_returns_self = mix.setBinaryInteraction(matrix) is mix

    bad_shape_raised = False
    try:
        mix.acentricFactors = [0.1, 0.2]
    except ValueError:
        bad_shape_raised = True

    mix.acentricFactors = [0.1, 0.2, 0.3]

    print("\nPhase 255: Mixture MRST CompositionalMixture API aliases")
    failures = 0
    failures += _check("getNumberOfComponents aliases component count",
                       mix.getNumberOfComponents() == mix.get_number_of_components() == mix.n == 3)
    failures += _check("setBinaryInteraction lower-triangular vector returns self",
                       vector_setter_returns_self)
    failures += _check("lower-triangular vector follows MRST ordering",
                       np.allclose(bic_after_vector, expected_vector_bic))
    failures += _check("setBinaryInteraction matrix returns self",
                       matrix_setter_returns_self)
    failures += _check("getBinaryInteraction aliases stored BIC matrix",
                       np.allclose(mix.getBinaryInteraction(), matrix)
                       and np.allclose(mix.get_binary_interaction(), matrix))
    failures += _check("acentricFactors aliases acentric with setter",
                       np.allclose(mix.acentricFactors, [0.1, 0.2, 0.3])
                       and np.allclose(mix.acentric, [0.1, 0.2, 0.3]))
    failures += _check("acentricFactors rejects wrong vector length",
                       bad_shape_raised)

    if failures:
        print(f"\nPhase 255 failed ({failures} checks).")
        return 1
    print("\nMixture now exposes MRST CompositionalMixture method/property aliases.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
