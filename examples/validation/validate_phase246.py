"""Phase 246: MRST-style separator multi-output unpacking parity."""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pymrst.modules.ad_core import BasicSeparator, EOSSeparator, SeparatorGroup  # noqa: E402
from pymrst.modules.compositional.fluids import get_benchmark_mixture  # noqa: E402
from pymrst.modules.compositional.properties.components import EquationOfStateComponent  # noqa: E402
from pymrst.utils.units import ATM  # noqa: E402


class MassModel:
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

    def get_number_of_phases(self):
        return self.n_phases


def _check(name: str, condition: bool) -> int:
    print(f"  {'ok  ' if condition else 'FAIL'} {name}")
    return 0 if condition else 1


def main() -> int:
    mass_model = MassModel()
    basic = BasicSeparator(pressure=ATM, T=300.0)
    mass_result = basic.separateComponentMassStream(mass_model, [np.array([2.0]), np.array([3.0])])
    phase_mass_streams, mass_fractions, mass_densities = mass_result

    mixture, _ = get_benchmark_mixture("simple")
    eos = EOSSeparator(pressure=ATM, T=300.0)
    mole_result = eos.separateComponentMoleStream(mixture, np.array([[1.0, 0.5, 0.25]]))
    mole_streams, mole_fractions, molar_densities = mole_result

    group = SeparatorGroup(eos)
    surface_result = group.computeSurfaceRates(mixture, np.array([[1.0, 0.5, 0.25]]))
    surface_rates, phase_streams, phase_fractions, surface_densities = surface_result

    tree = SeparatorGroup(
        eos,
        pressure=[2.0 * ATM, ATM],
        T=[320.0, 300.0],
        phase_destination=[[0, -1], [0, -1]],
    )
    streams, diagnostics = tree.computeSurfaceMoleRates(
        mixture,
        np.array([[1.0, 0.5, 0.25]]),
        return_separator_streams=True,
    )
    molestreams, sep_densities, sep_fractions = diagnostics

    print("\nPhase 246: MRST-style separator multi-output unpacking parity")
    failures = 0
    failures += _check("BasicSeparator mass result unpacks as MRST three-output call",
                       phase_mass_streams is mass_result.phase_mass_streams
                       and mass_fractions is mass_result.phase_mass_fractions
                       and mass_densities is mass_result.densities
                       and len(phase_mass_streams) == 2)
    failures += _check("EOSSeparator mole result unpacks as MRST three-output call",
                       mole_streams is mole_result.phase_mole_streams
                       and mole_fractions is mole_result.phase_mole_fractions
                       and molar_densities is mole_result.molar_densities
                       and len(mole_streams) == 2)
    failures += _check("SeparatorGroup surface result unpacks while preserving dataclass fields",
                       surface_rates is surface_result.surface_rates
                       and phase_streams is surface_result.phase_streams
                       and phase_fractions is surface_result.phase_mole_fractions
                       and surface_densities is surface_result.densities
                       and len(surface_rates) == 2)
    failures += _check("SeparatorStream diagnostics unpack as MRST stream triplet",
                       len(streams) == 2
                       and molestreams is diagnostics.molestreams
                       and sep_densities is diagnostics.densities
                       and sep_fractions is diagnostics.molefractions)
    failures += _check("unpacked numerical payloads remain finite",
                       all(np.all(np.isfinite(np.asarray(v))) for v in phase_mass_streams)
                       and all(np.all(np.isfinite(np.asarray(v))) for v in mole_streams)
                       and all(np.all(np.isfinite(np.asarray(v))) for v in surface_rates))

    if failures:
        print(f"\nPhase 246 failed ({failures} checks).")
        return 1
    print("\nSeparator result objects now support MRST-style multi-output unpacking.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
