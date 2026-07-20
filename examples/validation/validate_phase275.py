"""Phase 275: getPartialVolumes callable utility parity."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pymrst.modules.compositional.fluids import get_benchmark_mixture  # noqa: E402
from pymrst.modules.compositional.models.eos import EquationOfStateModel  # noqa: E402
from pymrst.modules.compositional.utils import getPartialVolumes, get_partial_volumes, list_utils  # noqa: E402


def _check(name: str, condition: bool) -> int:
    print(f"  {'ok  ' if condition else 'FAIL'} {name}")
    return 0 if condition else 1


class ModelAdapter:
    water = False

    def __init__(self, eos):
        self.EOSModel = eos
        self.T = 350.0
        self.operators = SimpleNamespace(pv=np.array([10.0, 20.0, 30.0]))


def main() -> int:
    mixture, _ = get_benchmark_mixture("simple")
    eos = EquationOfStateModel(None, mixture)
    model = ModelAdapter(eos)
    state = SimpleNamespace(
        pressure=np.array([1.0e5, 1.1e5, 1.2e5]),
        components=np.tile([0.5, 0.3, 0.2], (3, 1)),
        L=np.ones(3),
        rho=np.column_stack([np.array([700.0, 710.0, 720.0]), np.array([5.0, 6.0, 7.0])]),
        s=np.column_stack([np.ones(3), np.zeros(3)]),
    )
    V_num, d_num = getPartialVolumes(model, state, acc=[np.ones(3)] * 3, singlePhaseStrategy="numerical")
    V_ana, d_ana = get_partial_volumes(model, state, singlePhaseStrategy="analytical", singlePhaseDifferentiation="none")
    expected = (1.0 / state.rho[:, 0])[:, None] / mixture.molarMass[None, :]
    state_prev = SimpleNamespace(**state.__dict__)
    state_prev.w = V_num * 2.0
    state_prev.w_p = state.pressure - 10.0
    V_subset, d_subset = getPartialVolumes(model, state_prev, acc=None, subset=np.array([True, False, True]))
    catalog = list_utils()
    state_two = SimpleNamespace(**state.__dict__)
    state_two.pressure = np.array([5.0e6, 5.5e6, 6.0e6])
    state_two.T = np.full(3, 350.0)
    state_two.components = np.tile([0.5, 0.3, 0.2], (3, 1))
    state_two.L = np.array([0.5, 0.5, 0.5])
    V_eos, d_eos = getPartialVolumes(
        model,
        state_two,
        twoPhaseStrategy="eos",
        twoPhaseDifferentiation="eos",
        pressureEpsilon=1.0e4,
    )

    print("\nPhase 275: getPartialVolumes callable utility parity")
    failures = 0
    failures += _check("numerical numeric-accumulation branch returns unit weights and zero dVdp",
                       V_num.shape == (3, 3) and np.allclose(V_num, 1.0) and np.allclose(d_num, 0.0))
    failures += _check("analytical single-phase branch uses density and molar mass",
                       np.allclose(V_ana, expected) and np.allclose(d_ana, 0.0))
    failures += _check("subset and previous-weight numerical derivative are honored",
                       V_subset.shape == (2, 3) and d_subset.shape == (2, 3)
                       and np.all(np.isfinite(d_subset)))
    failures += _check("two-phase EOS branch returns finite mass partial volumes",
                       V_eos.shape == (3, 3)
                       and np.all(np.isfinite(V_eos))
                       and np.all(V_eos > 0.0))
    failures += _check("two-phase EOS pressure derivative branch is finite",
                       d_eos.shape == (3, 3) and np.all(np.isfinite(d_eos)))
    failures += _check("utility catalog marks getPartialVolumes EOS branch implemented",
                       catalog["getPartialVolumes"]["status"] == "implemented_eos_finite_difference"
                       and catalog["getPartialVolumes"]["python_module"] == "get_partial_volumes")

    if failures:
        print(f"\nPhase 275 failed ({failures} checks).")
        return 1
    print("\ngetPartialVolumes now includes numerical, single-phase analytical, and two-phase EOS finite-difference branches.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
