"""Phase 270: getEOSComponent utility parity."""

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
from pymrst.modules.compositional.properties.components import EquationOfStateComponent  # noqa: E402
from pymrst.modules.compositional.utils import getEOSComponent, get_eos_component, list_utils  # noqa: E402


def _check(name: str, condition: bool) -> int:
    print(f"  {'ok  ' if condition else 'FAIL'} {name}")
    return 0 if condition else 1


class SurfaceFluid:
    rhoWS = np.array([1000.0])


class ThreePhaseAdapter:
    def __init__(self, eos):
        self.EOSModel = eos
        self.fluid = SurfaceFluid()
        self.phase_names = ("water", "liquid", "vapor")
        self.n_phases = 3

    def getNumberOfPhases(self):
        return 3

    def getLiquidIndex(self):
        return 1

    def getVaporIndex(self):
        return 2

    def getNonEoSPhaseNames(self):
        return ((), ("water",))

    def getPhaseNames(self):
        return self.phase_names


def main() -> int:
    mixture, _ = get_benchmark_mixture("simple")
    eos = EquationOfStateModel(None, mixture)
    model = SimpleNamespace(EOSModel=eos, n_phases=2, phase_names=("liquid", "vapor"))
    three_phase = ThreePhaseAdapter(eos)

    methane = getEOSComponent(model, 101325.0, 288.15, mixture.names[0], 0)
    decane = get_eos_component(model, 101325.0, 288.15, mixture.names[-1], len(mixture.names) - 1)
    water_ordered = getEOSComponent(three_phase, 101325.0, 288.15, mixture.names[0], 1, mrst_indexing=True)
    catalog = list_utils()

    print("\nPhase 270: getEOSComponent utility parity")
    failures = 0
    failures += _check("factory returns EquationOfStateComponent objects",
                       isinstance(methane, EquationOfStateComponent)
                       and isinstance(decane, EquationOfStateComponent))
    failures += _check("component identity and molar mass follow EOS mixture",
                       methane.name == mixture.names[0]
                       and decane.name == mixture.names[-1]
                       and np.isclose(methane.molarMass, mixture.molarMass[0])
                       and np.isclose(decane.molarMass, mixture.molarMass[-1]))
    failures += _check("two-phase surface mass fractions are finite and normalized",
                       np.all(np.isfinite(methane.surfacePhaseMassFractions))
                       and np.all(np.isfinite(decane.surfacePhaseMassFractions))
                       and np.isclose(methane.surfacePhaseMassFractions.sum(), 1.0)
                       and np.isclose(decane.surfacePhaseMassFractions.sum(), 1.0))
    failures += _check("pure phase densities are positive for liquid and vapor slots",
                       np.all(methane.surfacePhaseDensityPure > 0.0)
                       and np.all(decane.surfacePhaseDensityPure > 0.0))
    failures += _check("model phase ordering and non-EOS surface density are honored",
                       water_ordered.componentIndex == 0
                       and np.isclose(water_ordered.surfacePhaseDensityPure[0], 1000.0)
                       and np.isclose(water_ordered.surfacePhaseMassFractions[1:].sum(), 1.0))
    failures += _check("utility catalog marks getEOSComponent implemented",
                       catalog["getEOSComponent"]["status"] == "implemented"
                       and catalog["getEOSComponent"]["python_module"] == "get_eos_component")

    if failures:
        print(f"\nPhase 270 failed ({failures} checks).")
        return 1
    print("\ngetEOSComponent now constructs MRST-style EquationOfStateComponent objects from EOS flashes.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
