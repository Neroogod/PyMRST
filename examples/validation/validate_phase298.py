"""Phase 298: MRST-name compatibility modules."""

from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pymrst.modules.compositional.fluids.cool_props_compositional_mixture import (  # noqa: E402
    CoolPropsCompositionalMixture,
    cool_props_compositional_mixture,
    coolprops_compositional_mixture,
)
from pymrst.modules.compositional.fluids.blackoil import (  # noqa: E402
    convertRSRVtoKValue,
    convert_rs_rv_to_k_value,
    convert_rsrv_to_k_value,
)
from pymrst.modules.compositional.models.overall.sequential import (  # noqa: E402
    PressureCompositionalModel,
    TransportCompositionalModel,
    pressureEquationCompositional,
    pressure_equation_compositional,
    transportEquationCompositional,
    transportEquationOverallComposition,
    transport_equation_compositional,
    transport_equation_overall_composition,
)
from pymrst.modules.compositional.model import OverallCompositionModel  # noqa: E402
from pymrst.modules.compositional.utils.get_default_flash_non_linear_solver import (  # noqa: E402
    getDefaultFlashNonLinearSolver,
    get_default_flash_non_linear_solver,
    get_default_flash_nonlinear_solver,
)


def _check(name: str, condition: bool) -> int:
    print(f"  {'ok  ' if condition else 'FAIL'} {name}")
    return 0 if condition else 1


class DummySequentialModel:
    def __init__(self):
        self.pressure_calls = []
        self.transport_calls = []

    def pressure_step(self, state, dt, *args, **kwargs):
        self.pressure_calls.append((state, dt, args, kwargs))
        return "pressure", state, dt

    def transport_step(self, state0, state, dt, *args, **kwargs):
        self.transport_calls.append((state0, state, dt, args, kwargs))
        return "transport", state0, state, dt


def validate_alias_imports() -> int:
    solver = getDefaultFlashNonLinearSolver("maxIterations", 17)
    failures = 0
    failures += _check("nonlinear/non_linear flash solver module aliases match",
                       get_default_flash_non_linear_solver is get_default_flash_nonlinear_solver
                       and solver.maxIterations == 17)
    failures += _check("CoolProps module exposes both naming styles",
                       cool_props_compositional_mixture is coolprops_compositional_mixture
                       and CoolPropsCompositionalMixture.__name__ == "CoolPropsCompositionalMixture")
    failures += _check("RSRV K-value conversion exposes MRST camelCase alias",
                       convertRSRVtoKValue is convert_rs_rv_to_k_value
                       and convert_rsrv_to_k_value is convert_rs_rv_to_k_value)
    failures += _check("sequential model MRST names point at native overall model",
                       PressureCompositionalModel is OverallCompositionModel
                       and TransportCompositionalModel is OverallCompositionModel)
    return failures


def validate_equation_dispatch() -> int:
    dummy = DummySequentialModel()
    p_out = pressureEquationCompositional(dummy, "state", 3.0, "arg", flag=True)
    p_out2 = pressure_equation_compositional(dummy, "state2", 4.0)
    t_out = transportEquationCompositional(dummy, "state0", "state1", 5.0, "arg")
    t_out2 = transport_equation_compositional(dummy, "state0b", "state1b", 6.0)
    t_out3 = transportEquationOverallComposition(dummy, "a", "b", 7.0)
    t_out4 = transport_equation_overall_composition(dummy, "c", "d", 8.0)
    failures = 0
    failures += _check("pressureEquationCompositional dispatches to pressure_step",
                       p_out[:3] == ("pressure", "state", 3.0)
                       and p_out2[:3] == ("pressure", "state2", 4.0)
                       and len(dummy.pressure_calls) == 2)
    failures += _check("transport equation aliases dispatch to transport_step",
                       t_out[:4] == ("transport", "state0", "state1", 5.0)
                       and t_out2[:4] == ("transport", "state0b", "state1b", 6.0)
                       and t_out3[:4] == ("transport", "a", "b", 7.0)
                       and t_out4[:4] == ("transport", "c", "d", 8.0)
                       and len(dummy.transport_calls) == 4)
    return failures


def main() -> int:
    print("\nPhase 298: MRST-name compatibility modules")
    failures = validate_alias_imports()
    failures += validate_equation_dispatch()
    if failures:
        print(f"\nPhase 298 failed ({failures} checks).")
        return 1
    print("\nMRST-name compatibility modules import and dispatch correctly.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
