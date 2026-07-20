"""Phase 284: getPartialVolumes analytical EOS-density fallback."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pymrst.modules.compositional.backend import backend_flow_props  # noqa: E402
from pymrst.modules.compositional.fluids import get_benchmark_mixture  # noqa: E402
from pymrst.modules.compositional.models.eos import EquationOfStateModel  # noqa: E402
from pymrst.modules.compositional.properties import molar_density  # noqa: E402
from pymrst.modules.compositional.utils import getPartialVolumes  # noqa: E402


def _check(name: str, condition: bool) -> int:
    print(f"  {'ok  ' if condition else 'FAIL'} {name}")
    return 0 if condition else 1


class ModelAdapter:
    water = False

    def __init__(self, eos):
        self.EOSModel = eos
        self.T = 350.0


def _expected_single_phase(mixture, eos, state, liquid: bool):
    p = np.asarray(state.pressure, dtype=float).reshape(-1)
    T = np.asarray(state.T, dtype=float).reshape(-1)
    z = np.asarray(state.components, dtype=float)
    z = z / z.sum(axis=1, keepdims=True)
    fp = backend_flow_props(mixture, None, p, T, z)
    if liquid:
        rho_molar = molar_density(mixture, p, T, fp.x, fp.Z_L)
        phase_z = fp.x
    else:
        rho_molar = molar_density(mixture, p, T, fp.y, fp.Z_V)
        phase_z = fp.y
    rho_mass = np.asarray(rho_molar, dtype=float).reshape(-1) * (np.asarray(phase_z, dtype=float) @ mixture.molarMass)
    return (1.0 / rho_mass)[:, None] / mixture.molarMass[None, :]


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
    expected_liq = _expected_single_phase(mixture, eos, liquid_state, liquid=True)
    expected_vap = _expected_single_phase(mixture, eos, vapor_state, liquid=False)

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
        "liquid single-phase analytical branch derives density from EOS when state.rho is absent",
        V_liq.shape == (3, 3) and np.allclose(V_liq, expected_liq, rtol=1.0e-10, atol=1.0e-14) and np.allclose(d_liq, 0.0),
    )
    failures += _check(
        "vapor single-phase analytical branch derives density from EOS when state.rho is absent",
        V_vap.shape == (3, 3) and np.allclose(V_vap, expected_vap, rtol=1.0e-10, atol=1.0e-14) and np.allclose(d_vap, 0.0),
    )
    failures += _check(
        "analytical EOS-density fallback honors subset selection",
        V_subset.shape == (2, 3) and d_subset.shape == (2, 3) and np.allclose(V_subset, expected_liq[subset]),
    )
    failures += _check(
        "legacy state.rho analytical branch still takes precedence over EOS-derived density",
        np.allclose(V_rho, expected_rho) and np.allclose(d_rho, 0.0) and not np.allclose(V_rho, expected_liq),
    )

    if failures:
        print(f"\nPhase 284 failed ({failures} checks).")
        return 1
    print("\ngetPartialVolumes analytical single-phase branch now derives density from EOS state when state.rho is absent.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
