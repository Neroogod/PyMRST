"""Phase 254: EquationOfStateComponent MRST property aliases."""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pymrst.modules.compositional.properties.components import EquationOfStateComponent  # noqa: E402
from pymrst.utils.units import ATM  # noqa: E402


def _check(name: str, condition: bool) -> int:
    print(f"  {'ok  ' if condition else 'FAIL'} {name}")
    return 0 if condition else 1


def main() -> int:
    component = EquationOfStateComponent(
        "Methane",
        pressure=ATM,
        T=300.0,
        component_index=0,
        surface_phase_mass_fractions=[0.2, 0.8],
        surface_phase_density_pure=[700.0, 5.0],
        molar_mass=0.016,
    )

    negative_index_raised = False
    try:
        component.componentIndex = -1
    except ValueError:
        negative_index_raised = True

    component.componentIndex = 1
    component.surfacePhaseMassFractions = [0.3, 0.7]
    component.surfacePhaseDensityPure = [650.0, 4.0]
    component.molarMass = 0.017

    print("\nPhase 254: EquationOfStateComponent MRST property aliases")
    failures = 0
    failures += _check("componentIndex aliases component_index",
                       component.componentIndex == component.component_index == 1)
    failures += _check("surfacePhaseMassFractions aliases surface_phase_mass_fractions",
                       np.allclose(component.surfacePhaseMassFractions, [0.3, 0.7])
                       and np.allclose(component.surface_phase_mass_fractions, [0.3, 0.7]))
    failures += _check("surfacePhaseDensityPure aliases surface_phase_density_pure",
                       np.allclose(component.surfacePhaseDensityPure, [650.0, 4.0])
                       and np.allclose(component.surface_phase_density_pure, [650.0, 4.0]))
    failures += _check("molarMass aliases molar_mass",
                       np.isclose(component.molarMass, 0.017)
                       and np.isclose(component.molar_mass, 0.017))
    failures += _check("surface methods read aliased property updates",
                       component.getPhaseCompositionSurface() == [0.3, 0.7]
                       and np.allclose(
                           [v.reshape(-1)[0] for v in component.getPurePhaseDensitySurface(ATM, 300.0)],
                           [650.0, 4.0],
                       ))
    failures += _check("negative componentIndex setter is rejected", negative_index_raised)

    if failures:
        print(f"\nPhase 254 failed ({failures} checks).")
        return 1
    print("\nEquationOfStateComponent now exposes MRST-style property aliases.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
