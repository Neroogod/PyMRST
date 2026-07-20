"""Phase 274: newtonFugacityEquilibrium utility parity."""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pymrst.modules.compositional.eos import fugacity_coefficients  # noqa: E402
from pymrst.modules.compositional.fluids import get_benchmark_mixture  # noqa: E402
from pymrst.modules.compositional.models.eos import EquationOfStateModel  # noqa: E402
from pymrst.modules.compositional.utils import (  # noqa: E402
    estimateEquilibriumWilson,
    list_utils,
    newtonFugacityEquilibrium,
    newton_fugacity_equilibrium,
    solveRachfordRiceVLE,
)


def _check(name: str, condition: bool) -> int:
    print(f"  {'ok  ' if condition else 'FAIL'} {name}")
    return 0 if condition else 1


def _fugacity_mismatch(mix, p, T, x, y):
    ln_l, _ = fugacity_coefficients(mix, p, T, x, vapour=False)
    ln_v, _ = fugacity_coefficients(mix, p, T, y, vapour=True)
    f_l = np.exp(ln_l) * p[:, None] * x
    f_v = np.exp(ln_v) * p[:, None] * y
    return np.abs(f_l / f_v - 1.0)


def main() -> int:
    mixture, _ = get_benchmark_mixture("simple")
    eos = EquationOfStateModel(None, mixture)
    p = np.array([5.0e6, 7.0e6])
    T = np.array([350.0, 360.0])
    z = np.array([[0.50, 0.30, 0.20], [0.55, 0.30, 0.15]])
    K = estimateEquilibriumWilson(eos, p, T) * np.array([[1.15, 0.9, 1.1], [0.95, 1.2, 0.85]])
    V = solveRachfordRiceVLE(z, K)
    L = 1.0 - V
    x0 = z / (L[:, None] + (1.0 - L)[:, None] * K)
    y0 = K * x0
    x0 = x0 / x0.sum(axis=1, keepdims=True)
    y0 = y0 / y0.sum(axis=1, keepdims=True)
    before = _fugacity_mismatch(mixture, p, T, x0, y0).max()

    x, y, K_new, Z_L, Z_V, L_new, values = newtonFugacityEquilibrium(eos, p, T, z, K, L)
    x2, y2, K2, ZL2, ZV2, L2, values2 = newton_fugacity_equilibrium(eos, p, T, z, K_new, L_new)
    after = _fugacity_mismatch(mixture, p, T, x, y).max()
    catalog = list_utils()

    print("\nPhase 274: newtonFugacityEquilibrium utility parity")
    failures = 0
    failures += _check("Newton step returns MRST output tuple shapes",
                       x.shape == z.shape and y.shape == z.shape and K_new.shape == z.shape
                       and Z_L.shape == p.shape and Z_V.shape == p.shape
                       and L_new.shape == p.shape and values.shape == z.shape)
    failures += _check("liquid/vapor compositions remain normalized and bounded",
                       np.all(x >= 0.0) and np.all(y >= 0.0)
                       and np.allclose(x.sum(axis=1), 1.0)
                       and np.allclose(y.sum(axis=1), 1.0)
                       and np.all((L_new >= 0.0) & (L_new <= 1.0)))
    failures += _check("one Newton step reduces fugacity-ratio mismatch",
                       after < before and np.all(np.isfinite(values)))
    failures += _check("alias can continue the Newton iteration",
                       values2.max() <= values.max() * 1.05
                       and np.all(np.isfinite(K2)) and np.all(np.isfinite(ZL2)) and np.all(np.isfinite(ZV2))
                       and np.all((L2 >= 0.0) & (L2 <= 1.0)))
    failures += _check("utility catalog marks newtonFugacityEquilibrium implemented",
                       catalog["newtonFugacityEquilibrium"]["status"] == "implemented_newton_flash_step"
                       and catalog["newtonFugacityEquilibrium"]["python_module"] == "newton_fugacity_equilibrium")

    if failures:
        print(f"\nPhase 274 failed ({failures} checks).")
        return 1
    print("\nnewtonFugacityEquilibrium now performs a real AD-linearized flash Newton update.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
