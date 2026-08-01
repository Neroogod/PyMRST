"""Phase 284: getPartialVolumes analytical EOS-density fallback."""

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
from pymrst.modules.compositional.utils import getPartialVolumes  # noqa: E402


def _check(name: str, condition: bool) -> int:
    print(f"  {'ok  ' if condition else 'FAIL'} {name}")
    return 0 if condition else 1


class ModelAdapter:
    water = False

    def __init__(self, eos):
        self.EOSModel = eos
        self.T = 350.0


def main() -> int:
    mixture, _ = get_benchmark_mixture("simple")
    eos = EquationOfStateModel(None, mixture)
    model = ModelAdapter(eos)
    pressure = np.array([5.0e6, 5.5e6, 6.0e6])
    temperature = np.full(3, 350.0)
    components = np.tile([0.5, 0.3, 0.2], (3, 1))

    liquid_state = SimpleNamespace(
        pressure=pressure,
        T=temperature,
        components=components,
        L=np.ones(3),
    )
    vapor_state = SimpleNamespace(
        pressure=pressure,
        T=temperature,
        components=components,
        L=np.zeros(3),
    )
    V_liq, d_liq = getPartialVolumes(
        model,
        liquid_state,
        singlePhaseStrategy="analytical",
        singlePhaseDifferentiation="none",
    )
    V_vap, d_vap = getPartialVolumes(
        model,
        vapor_state,
        singlePhaseStrategy="analytical",
        singlePhaseDifferentiation="none",
    )
    subset = np.array([True, False, True])
    V_subset, d_subset = getPartialVolumes(
        model,
        liquid_state,
        subset=subset,
        singlePhaseStrategy="analytical",
        singlePhaseDifferentiation="none",
    )

    rho_state = SimpleNamespace(
        pressure=pressure,
        T=temperature,
        components=components,
        L=np.ones(3),
        rho=np.column_stack([np.array([700.0, 710.0, 720.0]), np.array([5.0, 6.0, 7.0])]),
    )
    V_rho, d_rho = getPartialVolumes(
        model,
        rho_state,
        singlePhaseStrategy="analytical",
        singlePhaseDifferentiation="none",
    )
    expected_rho = (1.0 / rho_state.rho[:, 0])[:, None] / mixture.molarMass[None, :]

    print("\nPhase 284: getPartialVolumes analytical EOS-density fallback")
    failures = 0
    failures += _check(
        "liquid single-phase branch returns finite component-specific EOS partial volumes",
        V_liq.shape == (3, 3) and np.all(np.isfinite(V_liq))
        and np.ptp(V_liq[0]) > 0.0
        and np.allclose(d_liq, 0.0),
    )
    failures += _check(
        "vapor single-phase branch returns finite component-specific EOS partial volumes",
        V_vap.shape == (3, 3) and np.all(np.isfinite(V_vap))
        and np.ptp(V_vap[0]) > 0.0
        and np.allclose(d_vap, 0.0) and not np.allclose(V_vap, V_liq),
    )
    failures += _check(
        "analytical EOS-density fallback honors subset selection",
        V_subset.shape == (2, 3) and d_subset.shape == (2, 3)
        and np.allclose(V_subset, V_liq[subset]),
    )
    failures += _check(
        "legacy state.rho analytical branch still takes precedence over EOS-derived density",
        np.allclose(V_rho, expected_rho) and np.allclose(d_rho, 0.0)
        and not np.allclose(V_rho, V_liq),
    )

    if failures:
        print(f"\nPhase 284 failed ({failures} checks).")
        return 1
    print("\ngetPartialVolumes analytical single-phase branch now uses MRST-style cubic-EOS AD partial volumes.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
