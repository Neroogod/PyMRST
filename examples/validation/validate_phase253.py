"""Phase 253: EOSSeparator uses its own EOSModel mixture when provided."""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pymrst.modules.ad_core.models.facilities.separators import EOSSeparator  # noqa: E402
from pymrst.modules.compositional.eos import Mixture  # noqa: E402
from pymrst.modules.compositional.fluids import get_benchmark_mixture  # noqa: E402
from pymrst.utils.units import ATM  # noqa: E402


def _check(name: str, condition: bool) -> int:
    print(f"  {'ok  ' if condition else 'FAIL'} {name}")
    return 0 if condition else 1


class EOSModelHolder:
    def __init__(self, mixture):
        self.mixture = mixture


def _two_component_mixture() -> Mixture:
    return Mixture(
        names=("C1", "C10"),
        Tcrit=np.array([190.56, 617.70]),
        Pcrit=np.array([45.99e5, 21.10e5]),
        Vcrit=np.array([9.86e-5, 6.03e-4]),
        acentric=np.array([0.011, 0.49]),
        molarMass=np.array([0.01604, 0.14229]),
        name="phase253 two-component override mixture",
    )


def main() -> int:
    fallback_mix, _ = get_benchmark_mixture("simple")
    override_mix = _two_component_mixture()
    sep = EOSSeparator(pressure=ATM, T=300.0, EOSModel=EOSModelHolder(override_mix))
    stream = np.array([0.8, 0.2])
    result = sep.separateComponentMoleStream(fallback_mix, stream)

    unset = EOSSeparator(pressure=ATM, T=300.0)
    unset_result = unset.separateComponentMoleStream(fallback_mix, np.array([0.8, 0.1, 0.1]))

    wrong_size_raised = False
    try:
        sep.separateComponentMoleStream(fallback_mix, np.array([0.8, 0.1, 0.1]))
    except ValueError:
        wrong_size_raised = True

    print("\nPhase 253: EOSSeparator EOSModel mixture override")
    failures = 0
    failures += _check("EOSModel mixture controls accepted stream component count",
                       result.phase_mole_fractions[0].shape[1] == override_mix.n
                       and result.phase_mole_fractions[1].shape[1] == override_mix.n)
    failures += _check("unset EOSModel still uses incoming model/mixture",
                       unset_result.phase_mole_fractions[0].shape[1] == fallback_mix.n)
    failures += _check("EOSModel mixture rejects stream sized for fallback model",
                       wrong_size_raised)
    failures += _check("override flash conserves component mole stream",
                       np.allclose(
                           result.phase_mole_fractions[0] * result.phase_mole_streams[0][:, None]
                           + result.phase_mole_fractions[1] * result.phase_mole_streams[1][:, None],
                           stream.reshape(1, -1),
                           atol=1.0e-10,
                       ))

    if failures:
        print(f"\nPhase 253 failed ({failures} checks).")
        return 1
    print("\nEOSSeparator now honors its own EOSModel mixture when one is provided.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
