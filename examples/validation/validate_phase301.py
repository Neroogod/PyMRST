"""Phase 301: equationsNaturalVariables compatibility helper."""

from __future__ import annotations

from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pymrst.modules.compositional.examples.utils import setup_simple_compositional_example  # noqa: E402
from pymrst.modules.compositional.models.natvars import (  # noqa: E402
    NaturalVariablesProblem,
    equationsNaturalVariables,
    equations_natural_variables,
)


def _check(name: str, condition: bool) -> int:
    print(f"  {'ok  ' if condition else 'FAIL'} {name}")
    return 0 if condition else 1


def build_case():
    setup = setup_simple_compositional_example(use_natural=True, nx=4, nstep=1)
    state0 = setup.state
    state = state0.copy()
    state.flag = np.zeros(state.pressure.shape, dtype=np.int8)
    return setup.model, state0, state


def validate_equation_problem() -> int:
    model, state0, state = build_case()
    problem, out_state = equations_natural_variables(
        state0,
        state,
        model,
        10.0,
        SimpleNamespace(bc=None, src=None),
        iteration=3,
    )
    assembled = problem.assembleSystem()
    failures = 0
    failures += _check("equations_natural_variables returns NaturalVariablesProblem",
                       isinstance(problem, NaturalVariablesProblem)
                       and out_state is state
                       and problem.iterationNo == 3)
    failures += _check("problem carries MRST equation names and types",
                       "closure" in problem.equationNames
                       and all(name.startswith("fugacity_") for name in problem.equationNames[-4:-1])
                       and len(problem.types) == len(problem.equations))
    failures += _check("primary variables include pressure and phase variables",
                       problem.primaryVariables[0] == "pressure"
                       and "sl" in problem.primaryVariables
                       and "sv" in problem.primaryVariables
                       and any(name.startswith("v_") for name in problem.primaryVariables)
                       and any(name.startswith("w_") for name in problem.primaryVariables))
    failures += _check("assembleSystem creates finite residual and matrix",
                       assembled is problem
                       and problem.assembled
                       and problem.b is not None
                       and problem.A is not None
                       and np.all(np.isfinite(problem.b))
                       and problem.A.shape[0] == problem.b.size)
    failures += _check("problem norm is finite",
                       np.isfinite(problem.norm()))
    return failures


def validate_options_and_alias() -> int:
    model, state0, state = build_case()
    problem, _ = equationsNaturalVariables(
        state0,
        state,
        model,
        10.0,
        None,
        "resOnly",
        True,
        "reduceToPressure",
        True,
        "staticWells",
        True,
    )
    failures = 0
    failures += _check("camelCase equationsNaturalVariables alias works",
                       isinstance(problem, NaturalVariablesProblem))
    failures += _check("resOnly/staticWells/reduceToPressure options are recorded",
                       problem.reduceToPressure
                       and problem.metadata["res_only"]
                       and problem.metadata["static_wells"])
    failures += _check("staticWells removes well primary variables",
                       "qOs" not in problem.primaryVariables
                       and "qGs" not in problem.primaryVariables
                       and "bhp" not in problem.primaryVariables)
    return failures


def main() -> int:
    print("\nPhase 301: equationsNaturalVariables compatibility helper")
    failures = validate_equation_problem()
    failures += validate_options_and_alias()
    if failures:
        print(f"\nPhase 301 failed ({failures} checks).")
        return 1
    print("\nequationsNaturalVariables now exposes the MRST natural-variable equation surface.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
