"""Phase 272: validateCompositionalForces utility parity."""

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
from pymrst.modules.compositional.utils import (  # noqa: E402
    getEOSComponent,
    list_utils,
    validateCompositionalForces,
    validate_compositional_forces,
)


def _check(name: str, condition: bool) -> int:
    print(f"  {'ok  ' if condition else 'FAIL'} {name}")
    return 0 if condition else 1


class ModelAdapter:
    def __init__(self, eos, components):
        self.EOSModel = eos
        self.FacilityModel = SimpleNamespace(pressure=101325.0, T=288.15)
        self.Components = components
        self.n_phases = 2

    def getNumberOfPhases(self):
        return 2

    def getLiquidIndex(self):
        return 0

    def getVaporIndex(self):
        return 1

    def getSurfaceDensities(self):
        return np.array([700.0, 4.0])


def main() -> int:
    mixture, info = get_benchmark_mixture("simple")
    eos = EquationOfStateModel(None, mixture)
    base = SimpleNamespace(EOSModel=eos, n_phases=2, phase_names=("liquid", "vapor"))
    components = [getEOSComponent(base, 101325.0, 288.15, name, i) for i, name in enumerate(mixture.names)]
    model = ModelAdapter(eos, components)
    active = SimpleNamespace(name="INJ", status=True, components=np.asarray(info.injection, dtype=float))
    inactive = SimpleNamespace(name="SHUT", status=False, components=np.asarray(info.initial, dtype=float))
    preset = SimpleNamespace(name="PRESET", status=True, components=np.asarray(info.initial, dtype=float), rhoS=np.array([1.0, 2.0]))
    forces = SimpleNamespace(W=[active, inactive, preset])

    out = validateCompositionalForces(model, forces, 1)
    alias_out = validate_compositional_forces(model, SimpleNamespace(W=[]))
    catalog = list_utils()
    missing_component_raises = False
    try:
        validateCompositionalForces(model, SimpleNamespace(W=[SimpleNamespace(status=True)]), 1)
    except AssertionError:
        missing_component_raises = True

    print("\nPhase 272: validateCompositionalForces utility parity")
    failures = 0
    failures += _check("function mutates and returns the forces object",
                       out is forces and alias_out.W == [])
    failures += _check("active compositional well receives finite rhoS",
                       hasattr(active, "rhoS")
                       and np.asarray(active.rhoS).shape == (2,)
                       and np.all(np.asarray(active.rhoS) > 0.0))
    failures += _check("inactive wells are left untouched",
                       not hasattr(inactive, "rhoS"))
    failures += _check("existing rhoS is not overwritten",
                       np.allclose(preset.rhoS, [1.0, 2.0]))
    failures += _check("missing well components raises MRST-style assertion",
                       missing_component_raises)
    failures += _check("utility catalog marks validateCompositionalForces implemented",
                       catalog["validateCompositionalForces"]["status"] == "implemented_well_rhos_validation"
                       and catalog["validateCompositionalForces"]["python_module"] == "validate_compositional_forces")

    if failures:
        print(f"\nPhase 272 failed ({failures} checks).")
        return 1
    print("\nvalidateCompositionalForces now validates compositional wells and fills rhoS from EOS flashes.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
