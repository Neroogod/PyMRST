"""Phase 244: EquationOfStateComponent MRST method-name parity."""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pymrst.modules.compositional.fluids import get_benchmark_mixture  # noqa: E402
from pymrst.modules.compositional.properties.components import EquationOfStateComponent  # noqa: E402


class FakeModel:
    def __init__(self, mixture, density, mass_fractions):
        self.mix = mixture
        self._density = density
        self._mass_fractions = mass_fractions

    def getProps(self, state, name):
        if name == "Density":
            return self._density
        if name == "ComponentPhaseMassFractions":
            return self._mass_fractions
        raise KeyError(name)

    def getNumberOfPhases(self):
        return 2

    def getLiquidIndex(self):
        return 0

    def getVaporIndex(self):
        return 1


def _check(name: str, condition: bool) -> int:
    print(f"  {'ok  ' if condition else 'FAIL'} {name}")
    return 0 if condition else 1


def main() -> int:
    mixture, _ = get_benchmark_mixture("simple")
    mass_fractions = np.array([
        [0.20, 0.80],
        [0.70, 0.10],
        [0.10, 0.10],
    ])
    density = np.array([500.0, 10.0])
    component = EquationOfStateComponent(
        "Methane",
        pressure=101325.0,
        T=288.15,
        component_index=0,
        surface_phase_mass_fractions=[0.1, 0.9],
        surface_phase_density_pure=[450.0, 0.7],
        molar_mass=mixture.molarMass[0],
    )
    model = FakeModel(mixture, density, mass_fractions)
    direct_phase = component.get_phase_composition(mass_fractions)
    mrst_phase = component.getPhaseComposition(model, None)
    direct_density = component.get_component_density(density, mass_fractions)
    mrst_density = component.getComponentDensity(model, None)
    injected = component.getPhaseComponentFractionInjection(
        model,
        None,
        [{"components": [0.5, 0.3, 0.2]}, {"components": [0.0, 1.0, 0.0]}],
    )

    print("\nPhase 244: EquationOfStateComponent MRST method-name parity")
    failures = 0
    failures += _check("getPhaseComposition aliases Python phase composition",
                       all(np.allclose(a, b) for a, b in zip(direct_phase, mrst_phase)))
    failures += _check("getComponentDensity aliases Python component density",
                       all(np.allclose(a, b) for a, b in zip(direct_density, mrst_density)))
    failures += _check("getComponentDensity accepts extra.rho dictionary",
                       all(np.allclose(a, b) for a, b in zip(
                           direct_density,
                           component.getComponentDensity(model, None, extra={"rho": density}),
                       )))
    failures += _check("surface camelCase aliases remain available",
                       component.getPhaseCompositionSurface(model, None, 101325.0, 288.15) == [0.1, 0.9]
                       and len(component.getPurePhaseDensitySurface(model, None, 101325.0, 288.15)) == 2)
    failures += _check("getPhaseComponentFractionInjection returns liquid/vapour mass-fraction vectors",
                       len(injected) == 2
                       and injected[0].shape == (2,)
                       and injected[1].shape == (2,)
                       and np.allclose(injected[0], injected[1])
                       and np.all(injected[0] >= 0.0))

    if failures:
        print(f"\nPhase 244 failed ({failures} checks).")
        return 1
    print("\nEquationOfStateComponent now exposes MRST camelCase component APIs.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
