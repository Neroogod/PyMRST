"""Phase 282: EOSSeparator mass-stream separation."""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pymrst.modules.ad_core.models.facilities.separators import EOSSeparator, SeparatorGroup  # noqa: E402
from pymrst.modules.compositional.eos import simple_mixture  # noqa: E402
from pymrst.utils.units import ATM  # noqa: E402


def _check(name: str, condition: bool) -> int:
    print(f"  {'ok  ' if condition else 'FAIL'} {name}")
    return 0 if condition else 1


def main() -> int:
    print("\nPhase 282: EOSSeparator mass-stream separation")
    mix = simple_mixture()
    sep = EOSSeparator(pressure=ATM, T=300.0, EOSModel=mix)
    mass_stream = np.array([
        [0.016, 0.040, 0.012],
        [0.010, 0.055, 0.018],
    ])
    mole_stream = mass_stream / mix.molarMass[None, :]

    mass_split = sep.separateComponentMassStream(mix, mass_stream)
    mole_split = sep.separateComponentMoleStream(mix, mole_stream)
    group = SeparatorGroup(sep, mode="mass")
    surface = group.getSurfaceRates(mix, mass_stream)

    reconstructed_mass = mass_split.phase_mass_fractions[0] + mass_split.phase_mass_fractions[1]
    mole_to_mass_l = (
        mole_split.phase_mole_fractions[0]
        * mole_split.phase_mole_streams[0][:, None]
        * mix.molarMass[None, :]
    )
    mole_to_mass_v = (
        mole_split.phase_mole_fractions[1]
        * mole_split.phase_mole_streams[1][:, None]
        * mix.molarMass[None, :]
    )

    failures = 0
    failures += _check(
        "EOS mass split conserves component mass streams",
        np.allclose(reconstructed_mass, mass_stream, rtol=1.0e-11, atol=1.0e-14),
    )
    failures += _check(
        "EOS mass split is consistent with mole split converted back to mass",
        np.allclose(mass_split.phase_mass_fractions[0], mole_to_mass_l, rtol=1.0e-11, atol=1.0e-14)
        and np.allclose(mass_split.phase_mass_fractions[1], mole_to_mass_v, rtol=1.0e-11, atol=1.0e-14),
    )
    failures += _check(
        "EOS mass split reports finite positive phase densities where phase stream exists",
        all(
            np.all(rho[np.abs(q) > 0.0] > 0.0)
            and np.all(np.isfinite(rho[np.abs(q) > 0.0]))
            for q, rho in zip(mass_split.phase_mass_streams, mass_split.densities)
        ),
    )
    failures += _check(
        "SeparatorGroup mass mode now accepts EOSSeparator",
        len(surface.surface_rates) == 2
        and all(np.all(np.isfinite(q)) for q in surface.surface_rates)
        and all(np.all(q >= 0.0) for q in surface.surface_rates),
    )

    if failures:
        print(f"\nPhase 282 failed ({failures} checks).")
        return 1
    print("\nEOSSeparator now supports MRST-style component mass-stream separation.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
