"""Phase 330: EquilibriumConstantModel uses the base MRST K-value evaluator."""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pymrst.modules.compositional.eos import simple_mixture  # noqa: E402
from pymrst.modules.compositional.models.eos import EquilibriumConstantModel  # noqa: E402


def _check(name: str, condition: bool) -> int:
    print(f"  {'ok  ' if condition else 'FAIL'} {name}")
    return 0 if condition else 1


def validate_subclass_k_clipping_and_matrix_function() -> int:
    model = EquilibriumConstantModel(
        None,
        simple_mixture(),
        lambda P, T, z: np.column_stack([
            1.0e12 + 0.0 * P,
            1.0e-12 + 0.0 * P,
            2.0 + 0.0 * P,
        ]),
    )
    p = np.array([1.0e7, 1.2e7])
    T = np.array([300.0, 310.0])
    z = np.array([[0.35, 0.25, 0.40], [0.20, 0.35, 0.45]])
    K = model.evaluateEquilibriumConstants(p, T, z)

    failures = 0
    failures += _check("single matrix-valued K function is still accepted",
                       K.shape == z.shape and np.allclose(K[:, 2], 2.0))
    failures += _check("EquilibriumConstantModel now applies MRST K clipping",
                       np.allclose(K[:, 0], 1.0e8)
                       and np.allclose(K[:, 1], 1.0e-8))
    return failures


def validate_subclass_cell_output() -> int:
    model = EquilibriumConstantModel(
        None,
        simple_mixture(),
        [
            lambda P, T, z: 0.5 + 0.0 * P,
            lambda P, T, z: 1.2 + 0.0 * P,
            lambda P, T, z: 2.0 + 0.0 * P,
        ],
    )
    p = np.array([1.0e7, 1.2e7])
    T = np.array([300.0, 310.0])
    z_cols = [
        np.array([0.35, 0.20]),
        np.array([0.25, 0.35]),
        np.array([0.40, 0.45]),
    ]
    K = model.evaluate_equilibrium_constants(p, T, z_cols)

    failures = 0
    failures += _check("cell/list composition input returns MRST-style list output",
                       isinstance(K, list)
                       and len(K) == 3
                       and np.allclose(K[0], 0.5)
                       and np.allclose(K[1], 1.2)
                       and np.allclose(K[2], 2.0))
    return failures


def main() -> int:
    print("\nPhase 330: EquilibriumConstantModel uses the base MRST K-value evaluator")
    failures = 0
    failures += validate_subclass_k_clipping_and_matrix_function()
    failures += validate_subclass_cell_output()
    if failures:
        print(f"\nPhase 330 failed ({failures} checks).")
        return 1
    print("\nEquilibriumConstantModel base K-evaluator parity is validated.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
