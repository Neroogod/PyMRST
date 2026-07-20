"""Phase 283: mass-input EOS separator trees."""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pymrst.modules.ad_core.models.facilities.separators import EOSSeparator, SeparatorGroup  # noqa: E402
from pymrst.modules.compositional.eos import simple_mixture  # noqa: E402
from pymrst.utils.units import ATM, BARSA  # noqa: E402


def _check(name: str, condition: bool) -> int:
    print(f"  {'ok  ' if condition else 'FAIL'} {name}")
    return 0 if condition else 1


def _tree(separator, mode: str) -> SeparatorGroup:
    return SeparatorGroup(
        separator,
        pressure=np.array([60.0 * BARSA, 20.0 * BARSA]),
        T=np.array([340.0, 310.0]),
        phase_destination=np.array([[0, -1], [0, -1]]),
        mode=mode,
    )


def main() -> int:
    print("\nPhase 283: mass-input EOS separator trees")
    mix = simple_mixture()
    surface = EOSSeparator(pressure=ATM, T=288.15, EOSModel=mix)
    mass_stream = np.array([
        [0.016, 0.032, 0.020],
        [0.020, 0.020, 0.030],
    ])
    mole_stream = mass_stream / mix.molarMass[None, :]

    mass_tree = _tree(surface, "mass")
    mole_tree = _tree(surface, "moles")
    mass_result = mass_tree.getSurfaceRates(mix, mass_stream)
    mole_result = mole_tree.computeSurfaceRates(mix, mole_stream, mode="moles")

    final_component_mass = mass_result.phase_mole_fractions[0] + mass_result.phase_mole_fractions[1]
    mole_component_mass = [stream * mix.molarMass[None, :] for stream in mole_result.surface_mole_streams]

    failures = 0
    failures += _check(
        "mass-input separator tree conserves component mass",
        np.allclose(final_component_mass, mass_stream, rtol=1.0e-10, atol=1.0e-13),
    )
    failures += _check(
        "mass-input tree matches mole-input tree after unit conversion",
        all(np.allclose(a, b, rtol=1.0e-10, atol=1.0e-13)
            for a, b in zip(mass_result.phase_mole_fractions, mole_component_mass))
        and all(np.allclose(a, b, rtol=1.0e-10, atol=1.0e-13)
                for a, b in zip(mass_result.surface_rates, mole_result.surface_rates)),
    )
    failures += _check(
        "mass-input tree keeps separator stream diagnostics from mole routing",
        mass_result.separator_streams is not None
        and len(mass_result.separator_streams.molestreams) == 2
        and mass_result.surface_mole_streams is not None
        and all(stream.shape == mass_stream.shape for stream in mass_result.surface_mole_streams),
    )
    failures += _check(
        "mass-input tree reports finite stock-tank densities and rates",
        all(np.all(np.isfinite(rho[np.abs(q) > 0.0])) for q, rho in zip(mass_result.surface_rates, mass_result.densities))
        and all(np.all(q >= 0.0) for q in mass_result.surface_rates),
    )

    if failures:
        print(f"\nPhase 283 failed ({failures} checks).")
        return 1
    print("\nSeparatorGroup now routes mass-input EOS separator trees through the MRST-style mole flash path.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
