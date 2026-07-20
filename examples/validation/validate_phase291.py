"""Phase 291: BasicSeparator mole-stream split parity."""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pymrst.modules.ad_core.models.facilities import (  # noqa: E402
    BasicSeparator,
    SeparatorMoleResult,
)
from pymrst.modules.compositional.properties.components import EquationOfStateComponent  # noqa: E402
from pymrst.utils.units import ATM  # noqa: E402


def _check(name: str, condition: bool) -> int:
    print(f"  {'ok  ' if condition else 'FAIL'} {name}")
    return 0 if condition else 1


class FakeModel:
    n_phases = 2

    def __init__(self):
        self.components = [
            EquationOfStateComponent(
                "Light",
                pressure=ATM,
                T=300.0,
                component_index=0,
                surface_phase_mass_fractions=[0.2, 0.8],
                surface_phase_density_pure=[700.0, 5.0],
                molar_mass=0.016,
            ),
            EquationOfStateComponent(
                "Heavy",
                pressure=ATM,
                T=300.0,
                component_index=1,
                surface_phase_mass_fractions=[0.9, 0.1],
                surface_phase_density_pure=[850.0, 15.0],
                molar_mass=0.100,
            ),
        ]

    @property
    def molarMass(self):
        return np.array([c.molarMass for c in self.components])

    def get_number_of_phases(self):
        return self.n_phases


def validate_basic_separator_mole_split() -> int:
    model = FakeModel()
    sep = BasicSeparator(pressure=ATM, T=300.0)
    mass_stream = np.array([[2.0, 3.0], [4.0, 1.0]])
    mole_stream = mass_stream / model.molarMass[None, :]
    out = sep.separate_component_mole_stream(model, mole_stream)
    reconstructed_mass = np.zeros_like(mass_stream)
    for phase_total, phase_fraction in zip(out.phase_mole_streams, out.phase_mole_fractions):
        reconstructed_mass += phase_fraction * phase_total[:, None] * model.molarMass[None, :]

    failures = 0
    failures += _check("BasicSeparator returns a mole-result object",
                       isinstance(out, SeparatorMoleResult)
                       and len(out.phase_mole_streams) == 2)
    failures += _check("mole split preserves component mass after round trip",
                       np.allclose(reconstructed_mass, mass_stream))
    failures += _check("phase mole fractions are normalized for active streams",
                       np.allclose(out.phase_mole_fractions[0].sum(axis=1), [1.0, 1.0])
                       and np.allclose(out.phase_mole_fractions[1].sum(axis=1), [1.0, 1.0]))
    failures += _check("molar densities are finite for active streams",
                       np.all(np.isfinite(out.molar_densities[0]))
                       and np.all(np.isfinite(out.molar_densities[1]))
                       and np.all(out.molar_densities[0] > out.molar_densities[1]))
    failures += _check("mass_result is the underlying BasicSeparator mass split",
                       np.allclose(out.mass_result.phase_mass_streams[0], [3.1, 1.7])
                       and np.allclose(out.mass_result.phase_mass_streams[1], [1.9, 3.3]))
    return failures


def main() -> int:
    print("\nPhase 291: BasicSeparator mole-stream split")
    failures = validate_basic_separator_mole_split()
    if failures:
        print(f"\nPhase 291 failed ({failures} checks).")
        return 1
    print("\nBasicSeparator now supports MRST-style mole-stream separation.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
