"""Phase 256: CompositionalPropertyModel MRST method surface."""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pymrst.modules.compositional.eos import Mixture, OMEGA_B  # noqa: E402
from pymrst.modules.compositional.fluids import (  # noqa: E402
    CompositionalPropertyModel,
    mass_density,
    molar_density,
    viscosity_lbc,
)


class _EOS:
    def __init__(self, omegaB: float = OMEGA_B):
        self.omegaB = omegaB


def _check(name: str, condition: bool) -> int:
    print(f"  {'ok  ' if condition else 'FAIL'} {name}")
    return 0 if condition else 1


def _mixture() -> Mixture:
    return Mixture(
        names=("C1", "C3", "C10"),
        Tcrit=np.array([190.56, 369.83, 617.7]),
        Pcrit=np.array([4.5992e6, 4.248e6, 2.11e6]),
        Vcrit=np.array([9.86e-5, 2.00e-4, 6.17e-4]),
        acentric=np.array([0.0114, 0.1523, 0.4884]),
        molarMass=np.array([0.01604, 0.04410, 0.14229]),
    )


def main() -> int:
    mix = _mixture()
    p = np.array([1.1e7, 1.4e7])
    T = np.array([345.0, 365.0])
    x = np.array([[0.55, 0.25, 0.20], [0.30, 0.30, 0.40]])
    Z = np.array([0.72, 0.84])
    shift = np.array([0.02, -0.01, 0.04])
    eos = _EOS()
    custom_eos = _EOS(OMEGA_B * 1.07)

    model = CompositionalPropertyModel(mix)
    shifted = CompositionalPropertyModel(mixture=mix, volume_shift=shift)

    p_pc, t_pc, v_c, mw = model.computePseudoCriticalPhaseProperties(x)

    missing_fluid_raised = False
    try:
        CompositionalPropertyModel().computeDensity(eos, p, x, Z, T, True)
    except ValueError:
        missing_fluid_raised = True

    shifted.volume_shift = shift * 2.0

    print("\nPhase 256: CompositionalPropertyModel MRST method surface")
    failures = 0
    failures += _check("computeMolarDensity delegates to molar_density",
                       np.allclose(model.computeMolarDensity(eos, p, x, Z, T, True),
                                   molar_density(mix, p, T, x, Z)))
    failures += _check("computeDensity delegates to mass_density",
                       np.allclose(model.computeDensity(eos, p, x, Z, T, True),
                                   mass_density(mix, p, T, x, Z)))
    failures += _check("computeViscosity delegates to viscosity_lbc",
                       np.allclose(model.computeViscosity(eos, p, x, Z, T, True),
                                   viscosity_lbc(mix, p, T, x, Z)))
    failures += _check("custom EOS.omegaB is honored for volume-shifted density",
                       np.allclose(
                           shifted.computeMolarDensity(custom_eos, p, x, Z, T, True),
                           molar_density(mix, p, T, x, Z, shift * 2.0, omega_b=custom_eos.omegaB),
                       ))
    failures += _check("volume_shift aliases MRST volumeShift",
                       np.allclose(shifted.volumeShift, shift * 2.0)
                       and np.allclose(shifted.volume_shift, shift * 2.0))
    failures += _check("computePseudoCriticalPhaseProperties returns MRST order",
                       np.allclose(p_pc, x @ mix.Pcrit)
                       and np.allclose(t_pc, x @ mix.Tcrit)
                       and np.allclose(v_c, x @ mix.Vcrit)
                       and np.allclose(mw, x @ mix.molarMass))
    failures += _check("missing fluid raises before evaluation", missing_fluid_raised)

    if failures:
        print(f"\nPhase 256 failed ({failures} checks).")
        return 1
    print("\nCompositionalPropertyModel exposes MRST-style density/viscosity methods.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
