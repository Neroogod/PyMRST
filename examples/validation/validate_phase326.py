"""Phase 326: MRST EquationOfStateModel phase-flag/update helpers."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
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


def validate_composition_helpers() -> int:
    eos = EquationOfStateModel(None, _mixture(), "PR")
    L = np.array([1.0, 0.0, 0.35])
    K = np.array([
        [1.8, 0.7, 0.2],
        [1.7, 0.8, 0.25],
        [2.0, 0.75, 0.22],
    ])
    z = np.array([
        [0.80, 0.15, 0.05],
        [0.76, 0.18, 0.06],
        [0.70, 0.22, 0.08],
    ])

    liquid = eos.computeLiquid(L, K, z)
    vapor = eos.computeVapor(L, K, z)
    denom = L[:, None] + (1.0 - L[:, None]) * K
    ref_liquid = z / denom
    ref_liquid /= np.sum(ref_liquid, axis=1, keepdims=True)
    ref_vapor = K * z / denom
    ref_vapor /= np.sum(ref_vapor, axis=1, keepdims=True)

    failures = 0
    failures += _check("computeLiquid matches MRST Rachford-Rice composition formula",
                       np.allclose(liquid, ref_liquid))
    failures += _check("computeVapor matches MRST Rachford-Rice composition formula",
                       np.allclose(vapor, ref_vapor))
    failures += _check("state overloads are accepted for computeLiquid/computeVapor",
                       np.allclose(eos.computeLiquid({"L": L, "K": K, "components": z}), ref_liquid)
                       and np.allclose(eos.computeVapor({"L": L, "K": K, "components": z}), ref_vapor))
    return failures


def validate_flags_and_update() -> int:
    eos = EquationOfStateModel(None, _mixture(), "PR")
    components = np.array([
        [0.80, 0.15, 0.05],
        [0.76, 0.18, 0.06],
        [0.70, 0.22, 0.08],
    ])
    state = {
        "pressure": np.array([1.0e7, 1.1e7, 1.2e7]),
        "T": np.array([350.0, 355.0, 360.0]),
        "components": components,
        "L": np.array([1.0, 0.0, 0.35]),
        "K": np.array([
            [1.8, 0.7, 0.2],
            [1.7, 0.8, 0.25],
            [2.0, 0.75, 0.22],
        ]),
    }
    updated, report = eos.updateAfterConvergence(None, state, 86400.0)
    is_liquid, is_vapor, is_2ph = eos.getFlag(updated)

    failures = 0
    failures += _check("setFlag labels pure liquid, pure vapor and two-phase cells",
                       np.array_equal(updated["flag"], np.array([1, 2, 0]))
                       and np.array_equal(is_liquid, np.array([True, False, False]))
                       and np.array_equal(is_vapor, np.array([False, True, False]))
                       and np.array_equal(is_2ph, np.array([False, False, True])))
    failures += _check("getSinglePhase/getTwoPhaseFlag mirror MRST flag semantics",
                       np.array_equal(eos.getSinglePhase(updated), np.array([True, True, False]))
                       and np.array_equal(eos.getTwoPhaseFlag(updated), np.array([False, False, True])))
    failures += _check("updateAfterConvergence fills missing phase compositions",
                       np.allclose(updated["x"][0], components[0])
                       and np.allclose(updated["y"][0], components[0])
                       and np.allclose(updated["x"][1], components[1])
                       and np.allclose(updated["y"][1], components[1])
                       and updated["x"].shape == components.shape
                       and updated["y"].shape == components.shape
                       and isinstance(report, dict))

    obj = SimpleNamespace(L=np.array([True, False]), components=components[:2], K=state["K"][:2])
    eos.setFlag(obj, pureLiquid=np.array([True, True]), pureVapor=np.array([True, False]))
    failures += _check("explicit pure liquid mask dominates pure vapor mask",
                       np.array_equal(obj.flag, np.array([1, 1])))
    return failures


def main() -> int:
    print("\nPhase 326: MRST EquationOfStateModel phase-flag/update helpers")
    failures = 0
    failures += validate_composition_helpers()
    failures += validate_flags_and_update()
    if failures:
        print(f"\nPhase 326 failed ({failures} checks).")
        return 1
    print("\nEOS phase-flag/update helper parity is validated.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
