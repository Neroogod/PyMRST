"""Phase 292: separator mode aliases."""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pymrst.modules.ad_core.models.facilities import (  # noqa: E402
    BasicSeparator,
    EOSSeparator,
    SeparatorGroup,
)
from pymrst.modules.compositional.eos import simple_mixture  # noqa: E402
from pymrst.modules.compositional.properties.components import EquationOfStateComponent  # noqa: E402
from pymrst.utils.units import ATM  # noqa: E402


def _check(name: str, condition: bool) -> int:
    print(f"  {'ok  ' if condition else 'FAIL'} {name}")
    return 0 if condition else 1


class BasicModel:
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


def validate_mole_alias() -> int:
    mix = simple_mixture()
    group = SeparatorGroup(EOSSeparator(pressure=ATM, T=300.0), mode="mole")
    out = group.get_surface_rates(mix, np.array([0.02, 0.01, 0.03]))
    failures = 0
    failures += _check("SeparatorGroup accepts singular mole mode alias",
                       len(out.surface_rates) == 2
                       and out.surface_mole_streams is not None
                       and np.all(np.isfinite(out.surface_rates[0]))
                       and np.all(np.isfinite(out.surface_rates[1])))
    return failures


def validate_mass_aliases() -> int:
    model = BasicModel()
    group = SeparatorGroup(BasicSeparator(pressure=ATM, T=300.0), mode="masses")
    out = group.get_surface_rates(model, np.array([2.0, 3.0]))
    direct = group.compute_surface_rates(model, np.array([2.0, 3.0]), mode="kg")
    failures = 0
    failures += _check("SeparatorGroup accepts plural mass mode alias",
                       len(out.surface_rates) == 2
                       and out.molar_densities is None)
    failures += _check("explicit kg alias follows mass path",
                       np.allclose(direct.phase_streams[0], out.phase_streams[0])
                       and np.allclose(direct.phase_streams[1], out.phase_streams[1]))
    return failures


def main() -> int:
    print("\nPhase 292: separator mode aliases")
    failures = validate_mole_alias()
    failures += validate_mass_aliases()
    if failures:
        print(f"\nPhase 292 failed ({failures} checks).")
        return 1
    print("\nSeparatorGroup now accepts common MRST-style mode aliases.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
