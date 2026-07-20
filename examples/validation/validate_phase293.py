"""Phase 293: MRST-layout EquilibriumConstantModel."""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pymrst.modules.compositional.eos import simple_mixture, R_GAS  # noqa: E402
from pymrst.modules.compositional.models.eos import EquilibriumConstantModel  # noqa: E402


def _check(name: str, condition: bool) -> int:
    print(f"  {'ok  ' if condition else 'FAIL'} {name}")
    return 0 if condition else 1


class ConstantPropertyModel:
    """Small MRST-compatible property model for deterministic validation."""

    def computeMolarDensity(self, eos, p, x, Z, T, isLiquid):
        del eos, x, Z, T
        p = np.asarray(p, dtype=float).reshape(-1)
        return np.full(p.shape, 800.0 if isLiquid else 20.0)


def build_model() -> EquilibriumConstantModel:
    mix = simple_mixture()
    functions = [
        lambda P, T, z: np.full_like(P, 0.5, dtype=float),
        lambda P, T, z: np.full_like(P, 1.2, dtype=float),
        lambda P, T, z: np.full_like(P, 2.0, dtype=float),
    ]
    model = EquilibriumConstantModel(None, mix, functions)
    model.PropertyModel = ConstantPropertyModel()
    return model


def validate_k_functions() -> int:
    model = build_model()
    p = np.array([1.0e7, 1.2e7])
    T = np.array([300.0, 310.0])
    z = np.array([[0.35, 0.25, 0.40], [0.20, 0.35, 0.45]])
    K = model.evaluateEquilibriumConstants(p, T, z)

    matrix_model = EquilibriumConstantModel(
        None,
        simple_mixture(),
        lambda P, T, z: np.column_stack([0.5 + 0.0 * P, 1.2 + 0.0 * P, 2.0 + 0.0 * P]),
    )
    matrix_K = matrix_model.evaluate_equilibrium_constants(p, T, z)

    failures = 0
    failures += _check("per-component K functions return MRST component columns",
                       np.allclose(K, np.array([[0.5, 1.2, 2.0], [0.5, 1.2, 2.0]])))
    failures += _check("single matrix-valued K function is accepted",
                       np.allclose(matrix_K, K))
    failures += _check("pythonic equilibrium function alias is mutable",
                       len(model.equilibrium_constant_functions) == 3)
    return failures


def validate_fugacity_and_flash() -> int:
    model = build_model()
    p = np.array([1.0e7, 1.2e7])
    T = np.array([300.0, 310.0])
    z = np.array([[0.35, 0.25, 0.40], [0.20, 0.35, 0.45]])
    x = np.array([[0.50, 0.25, 0.25], [0.30, 0.30, 0.40]])
    y = np.array([[0.20, 0.30, 0.50], [0.15, 0.35, 0.50]])

    Z_L, Z_V, f_L, f_V = model.getCompressibilityAndFugacity(p, T, x, y, z)
    expected_Z_L = p / (800.0 * R_GAS * T)
    expected_Z_V = p / (20.0 * R_GAS * T)
    expected_K = np.array([[0.5, 1.2, 2.0], [0.5, 1.2, 2.0]])
    flash = model.flash(p, T, z)

    failures = 0
    failures += _check("compressibility factors use PropertyModel molar density",
                       np.allclose(Z_L, expected_Z_L) and np.allclose(Z_V, expected_Z_V))
    failures += _check("K-value fugacity follows MRST f_L=x*K, f_V=y",
                       np.allclose(f_L, x * expected_K) and np.allclose(f_V, y))
    failures += _check("flash returns normalized phase compositions",
                       np.allclose(flash.x.sum(axis=1), 1.0)
                       and np.allclose(flash.y.sum(axis=1), 1.0)
                       and np.all(flash.V >= 0.0)
                       and np.all(flash.V <= 1.0))
    failures += _check("flash carries K and compressibility outputs",
                       np.allclose(flash.K, expected_K)
                       and np.all(np.isfinite(flash.Z_L))
                       and np.all(np.isfinite(flash.Z_V)))
    return failures


def validate_state_update() -> int:
    model = build_model()
    state = {
        "pressure": np.array([1.0e7, 1.1e7]),
        "T": np.array([300.0, 305.0]),
        "components": np.array([[0.35, 0.25, 0.40], [0.20, 0.35, 0.45]]),
    }
    updated, report = model.updateAfterConvergence(None, state, 1.0, None)
    failures = 0
    failures += _check("updateAfterConvergence returns state and report",
                       updated is state and report is None)
    failures += _check("state update writes MRST x/y/Z_L/Z_V fields",
                       all(key in updated for key in ("x", "y", "Z_L", "Z_V", "L", "V")))
    failures += _check("updated phase compositions remain normalized",
                       np.allclose(updated["x"].sum(axis=1), 1.0)
                       and np.allclose(updated["y"].sum(axis=1), 1.0))
    return failures


def main() -> int:
    print("\nPhase 293: MRST-layout EquilibriumConstantModel")
    failures = validate_k_functions()
    failures += validate_fugacity_and_flash()
    failures += validate_state_update()
    if failures:
        print(f"\nPhase 293 failed ({failures} checks).")
        return 1
    print("\nMRST-layout EquilibriumConstantModel is available under models/eos.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
