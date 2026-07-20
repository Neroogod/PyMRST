"""Phase 325: MRST EquationOfStateModel.validateState parity."""

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
from pymrst.modules.compositional.utils import estimateEquilibriumWilson  # noqa: E402


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


def validate_missing_fields() -> int:
    eos = EquationOfStateModel(None, _mixture(), "PR")
    pressure = np.array([1.0e7, 1.2e7])
    temperature = np.array([350.0, 360.0])
    components = np.array([[0.80, 0.15, 0.05], [0.76, 0.18, 0.06]])
    state = {"pressure": pressure, "T": temperature, "components": components}

    validated = eos.validateState(state)
    expected_K = estimateEquilibriumWilson(eos, pressure, temperature)

    failures = 0
    failures += _check("validateState mutates and returns the input state",
                       validated is state)
    failures += _check("missing L is initialized to NaN column values",
                       state["L"].shape == (2,) and np.all(np.isnan(state["L"])))
    failures += _check("missing K is estimated from Wilson correlation",
                       state["K"].shape == components.shape
                       and np.allclose(state["K"], expected_K))
    failures += _check("missing Z fields are initialized to one",
                       np.allclose(state["Z_L"], np.ones(2))
                       and np.allclose(state["Z_V"], np.ones(2)))
    failures += _check("missing x/y fields copy overall composition",
                       np.allclose(state["x"], components)
                       and np.allclose(state["y"], components)
                       and state["x"] is not components
                       and state["y"] is not components)
    return failures


def validate_existing_xy() -> int:
    eos = EquationOfStateModel(None, _mixture(), "PR")
    x = np.array([[0.70, 0.20, 0.10], [0.66, 0.24, 0.10]])
    y = np.array([[0.90, 0.08, 0.02], [0.88, 0.10, 0.02]])
    state = {
        "pressure": np.array([1.0e7, 1.2e7]),
        "T": np.array([350.0, 360.0]),
        "components": np.array([[0.80, 0.15, 0.05], [0.76, 0.18, 0.06]]),
        "x": x,
        "y": y,
    }

    eos.validate_state(state)
    with np.errstate(divide="ignore", invalid="ignore"):
        expected_K = y / x

    failures = 0
    failures += _check("existing x/y define missing K by elementwise y/x",
                       np.allclose(state["K"], expected_K))
    failures += _check("existing x/y arrays are preserved",
                       state["x"] is x and state["y"] is y)
    return failures


def validate_object_state() -> int:
    eos = EquationOfStateModel(None, _mixture(), "PR")
    state = SimpleNamespace(
        pressure=np.array([1.0e7]),
        temperature=np.array([350.0]),
        components=np.array([[0.80, 0.15, 0.05]]),
    )
    validated = eos.validateState(state)

    failures = 0
    failures += _check("object-style states are supported",
                       validated is state
                       and hasattr(state, "K")
                       and hasattr(state, "L")
                       and hasattr(state, "Z_L")
                       and hasattr(state, "Z_V"))
    failures += _check("temperature alias is accepted when T is absent",
                       state.K.shape == (1, 3) and np.all(np.isfinite(state.K)))
    return failures


def main() -> int:
    print("\nPhase 325: MRST EquationOfStateModel.validateState parity")
    failures = 0
    failures += validate_missing_fields()
    failures += validate_existing_xy()
    failures += validate_object_state()
    if failures:
        print(f"\nPhase 325 failed ({failures} checks).")
        return 1
    print("\nEOS validateState parity is validated.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
