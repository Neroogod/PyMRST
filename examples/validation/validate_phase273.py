"""Phase 273: checkComponentMassBalance diagnostic parity."""

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
from pymrst.modules.compositional.utils import checkComponentMassBalance, check_component_mass_balance, list_utils  # noqa: E402


def _check(name: str, condition: bool) -> int:
    print(f"  {'ok  ' if condition else 'FAIL'} {name}")
    return 0 if condition else 1


class ModelAdapter:
    water = False

    def __init__(self, eos):
        self.EOSModel = eos
        self.operators = SimpleNamespace(pv=np.array([10.0, 20.0]))

    def getComponentNames(self):
        return self.EOSModel.getComponentNames()


def _state(x, y, s, rho, pressure=None, well_components=None):
    if pressure is None:
        pressure = np.array([1.0e5, 1.1e5])
    wellSol = [] if well_components is None else [SimpleNamespace(components=np.asarray(well_components, dtype=float))]
    return SimpleNamespace(
        pressure=np.asarray(pressure, dtype=float),
        x=np.asarray(x, dtype=float),
        y=np.asarray(y, dtype=float),
        s=np.asarray(s, dtype=float),
        rho=np.asarray(rho, dtype=float),
        wellSol=wellSol,
    )


def main() -> int:
    mixture, _ = get_benchmark_mixture("simple")
    eos = EquationOfStateModel(None, mixture)
    model = ModelAdapter(eos)
    state0 = _state(
        x=[[0.8, 0.2, 0.0], [0.7, 0.3, 0.0]],
        y=[[0.5, 0.4, 0.1], [0.4, 0.5, 0.1]],
        s=[[0.6, 0.4], [0.7, 0.3]],
        rho=[[700.0, 5.0], [720.0, 6.0]],
    )
    state1 = _state(
        x=[[0.75, 0.25, 0.0], [0.65, 0.35, 0.0]],
        y=[[0.45, 0.45, 0.1], [0.35, 0.55, 0.1]],
        s=[[0.55, 0.45], [0.65, 0.35]],
        rho=[[710.0, 5.2], [730.0, 6.2]],
        well_components=[[2.0, -1.0, 0.0]],
    )
    state2 = _state(
        x=[[0.70, 0.30, 0.0], [0.60, 0.40, 0.0]],
        y=[[0.40, 0.50, 0.1], [0.30, 0.60, 0.1]],
        s=[[0.50, 0.50], [0.60, 0.40]],
        rho=[[720.0, 5.5], [740.0, 6.5]],
        well_components=[[-0.5, 3.0, -1.0]],
    )
    schedule = SimpleNamespace(step=SimpleNamespace(val=np.array([2.0, 3.0])))

    report = checkComponentMassBalance(model, state0, [state1, state2], schedule, verbose=False)
    alias_report = check_component_mass_balance(model, state0, [state1], schedule, n=1, verbose=False)
    catalog = list_utils()

    print("\nPhase 273: checkComponentMassBalance diagnostic parity")
    failures = 0
    failures += _check("diagnostic returns component rows and aggregate total",
                       len(report["components"]) == 3 and set(report["total"]) >= {"start_total", "end_total"})
    failures += _check("well component streams are time-integrated by sign",
                       np.isclose(report["components"][0]["injected"], 4.0)
                       and np.isclose(report["components"][0]["produced"], 1.5)
                       and np.isclose(report["components"][1]["injected"], 9.0)
                       and np.isclose(report["components"][1]["produced"], 2.0))
    failures += _check("phase masses are finite positive diagnostics",
                       report["components"][0]["oil_start"] > 0.0
                       and report["components"][1]["gas_end"] > 0.0
                       and np.isfinite(report["total"]["end_total"]))
    failures += _check("alias and n argument limit evaluated states",
                       len(alias_report["components"]) == 3
                       and np.isclose(alias_report["components"][0]["injected"], 4.0))
    failures += _check("utility catalog marks checkComponentMassBalance implemented",
                       catalog["checkComponentMassBalance"]["status"] == "implemented_diagnostic_core"
                       and catalog["checkComponentMassBalance"]["python_module"] == "check_component_mass_balance")

    if failures:
        print(f"\nPhase 273 failed ({failures} checks).")
        return 1
    print("\ncheckComponentMassBalance now reports component storage and well-stream mass diagnostics.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
