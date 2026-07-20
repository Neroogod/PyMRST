"""Phase 299: natural-variable sequential compatibility layout."""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import scipy.sparse as sp

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pymrst.modules.compositional.examples.utils import setup_simple_compositional_example  # noqa: E402
from pymrst.modules.compositional.models.natvars import (  # noqa: E402
    PressureNaturalVariablesModel,
    PressureReducedLinearSystem,
    ReducedLinearizedSystem,
    TransportNaturalVariablesModel,
)
from pymrst.modules.compositional.models.natvars.sequential import (  # noqa: E402
    transportEquationsNaturalVars,
    transport_equations_natural_vars,
)


def _check(name: str, condition: bool) -> int:
    print(f"  {'ok  ' if condition else 'FAIL'} {name}")
    return 0 if condition else 1


def validate_reduced_linearized_system() -> int:
    A = sp.csr_matrix([[4.0, 1.0], [2.0, 3.0]])
    b = np.array([1.0, 2.0])
    full = np.linalg.solve(A.toarray(), b)
    system = ReducedLinearizedSystem(A=A, b=b, keepNum=1).assembleSystem()
    reduced = np.linalg.solve(system.A.toarray(), system.b)
    recovered, report = system.processResultAfterSolve(reduced, {"ok": True})
    pressure = PressureReducedLinearSystem(A=A, b=b, keepNum=1)
    pressure.assembleSystem()
    failures = 0
    failures += _check("ReducedLinearizedSystem forms Schur complement",
                       np.allclose(system.A.toarray(), [[4.0 - 1.0 * 2.0 / 3.0]])
                       and np.allclose(system.b, [1.0 - 2.0 / 3.0]))
    failures += _check("ReducedLinearizedSystem recovers eliminated variables",
                       np.allclose(recovered, full) and report == {"ok": True})
    failures += _check("PressureReducedLinearSystem assembles once and tracks flag",
                       pressure.assembled and np.allclose(pressure.A.toarray(), system.A.toarray()))
    return failures


def build_models():
    setup = setup_simple_compositional_example(use_natural=False, nx=4, nstep=1)
    pressure = PressureNaturalVariablesModel(
        setup.grid,
        setup.rock,
        setup.mixture,
        temperature=setup.info.T,
        wells=setup.wells,
        rho_s=(800.0, 10.0),
    )
    transport = TransportNaturalVariablesModel(
        setup.grid,
        setup.rock,
        setup.mixture,
        temperature=setup.info.T,
        wells=setup.wells,
        rho_s=(800.0, 10.0),
    )
    state = pressure.initial_state(setup.info.pressure, setup.info.initial)
    return pressure, transport, state


def validate_sequential_models() -> int:
    pressure, transport, state = build_models()
    transport_out = transport_equations_natural_vars(transport, state, state.copy(), 1.0, None)
    transport_out2 = transportEquationsNaturalVars(transport, state, state.copy(), 1.0, None)
    state.switchCount = np.ones(state.pressure.shape, dtype=int)
    updated, report = pressure.updateAfterConvergence(None, state, 1.0, None)
    failures = 0
    failures += _check("PressureNaturalVariablesModel sets MRST pressure flags",
                       pressure.useIncTolPressure
                       and pressure.useIncTolComposition
                       and pressure.allowLargeSaturations
                       and pressure.outputFluxes
                       and pressure.maxPhaseChangesNonLinear == 20
                       and pressure.singlePhaseStrategy == "numerical"
                       and pressure.twoPhaseStrategy == "numerical")
    failures += _check("TransportNaturalVariablesModel sets MRST transport flags",
                       transport.staticUpwind is False
                       and transport.upwindType == "potential"
                       and transport.allowLargeSaturations
                       and np.isinf(transport.maxPhaseChangesNonLinear))
    failures += _check("transport equation helpers return compatibility tuple",
                       hasattr(transport_out[0], "pressure")
                       and hasattr(transport_out2[0], "pressure")
                       and transport_out[0].pressure.shape == state.pressure.shape
                       and transport_out2[0].pressure.shape == state.pressure.shape)
    failures += _check("pressure updateAfterConvergence stores switchCount_p",
                       updated is state and report is None and np.array_equal(state.switchCount_p, state.switchCount))
    return failures


def main() -> int:
    print("\nPhase 299: natural-variable sequential compatibility layout")
    failures = validate_reduced_linearized_system()
    failures += validate_sequential_models()
    if failures:
        print(f"\nPhase 299 failed ({failures} checks).")
        return 1
    print("\nNatural-variable sequential compatibility layout is available.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
