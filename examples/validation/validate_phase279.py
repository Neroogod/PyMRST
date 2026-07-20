"""Phase 279: MRST-style BlackOilPropertyModel parity."""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pymrst.modules.compositional.fluids import BlackOilPropertyModel, list_fluids  # noqa: E402
from pymrst.modules.compositional.fluids.blackoil import BlackOilPropertyModel as BlackOilPropertyModelFromSubmodule  # noqa: E402


def _check(name: str, condition: bool) -> int:
    print(f"  {'ok  ' if condition else 'FAIL'} {name}")
    return 0 if condition else 1


def main() -> int:
    p = np.array([1.0e5, 2.0e5, 3.0e5])
    T = np.array([300.0, 310.0, 320.0])
    x = np.column_stack([np.array([0.2, 0.4, 0.6]), np.array([0.8, 0.6, 0.4])])

    rho_funcs = [
        lambda pp, tt, xx: 700.0 + 1.0e-5 * pp + 10.0 * np.asarray(xx[0]),
        lambda pp, tt, xx: 50.0 + 2.0e-5 * pp + 5.0 * np.asarray(xx[1]),
    ]
    mu_funcs = [
        lambda pp, tt, xx: 1.0e-3 + 1.0e-10 * pp + 1.0e-4 * np.asarray(xx[0]),
        lambda pp, tt, xx: 2.0e-5 + 1.0e-11 * pp + 1.0e-5 * np.asarray(xx[1]),
    ]
    stable = lambda pp, tt, zz: np.asarray(zz)[:, 0] > 0.5
    model = BlackOilPropertyModel(rho_funcs, mu_funcs, stable)
    rho_l = model.computeDensity(None, p, x, None, T, True)
    rho_g = model.computeDensity(None, p, x, None, T, False)
    rho_m = model.computeMolarDensity(None, p, [x[:, 0], x[:, 1]], None, T, True)
    mu_l = model.computeViscosity(None, p, x, None, T, True)
    stable_out = model.checkStability(p, T, x)
    catalog = list_fluids()

    print("\nPhase 279: MRST-style BlackOilPropertyModel parity")
    failures = 0
    failures += _check(
        "top-level and blackoil submodule export the same class",
        BlackOilPropertyModel is BlackOilPropertyModelFromSubmodule,
    )
    failures += _check(
        "computeDensity dispatches liquid/vapour function surfaces",
        np.allclose(rho_l, 700.0 + 1.0e-5 * p + 10.0 * x[:, 0])
        and np.allclose(rho_g, 50.0 + 2.0e-5 * p + 5.0 * x[:, 1]),
    )
    failures += _check(
        "computeMolarDensity mirrors black-oil density convention",
        np.allclose(rho_m, rho_l),
    )
    failures += _check(
        "computeViscosity dispatches function surfaces",
        np.allclose(mu_l, 1.0e-3 + 1.0e-10 * p + 1.0e-4 * x[:, 0]),
    )
    failures += _check(
        "checkStability delegates optional stability function",
        np.array_equal(stable_out, np.array([False, False, True])),
    )
    failures += _check(
        "fluid catalog marks BlackOilPropertyModel implemented",
        catalog["BlackOilPropertyModel"]["status"] == "implemented"
        and catalog["BlackOilPropertyModel"]["python_module"] == "black_oil_property_model",
    )

    if failures:
        print(f"\nPhase 279 failed ({failures} checks).")
        return 1
    print("\nBlackOilPropertyModel now supports MRST-style function-backed construction and camelCase property methods.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
