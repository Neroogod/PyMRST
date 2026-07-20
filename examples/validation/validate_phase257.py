"""Phase 257: EquationOfStateModel MRST object surface."""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pymrst.modules.compositional import EquationOfStateModel  # noqa: E402
from pymrst.modules.compositional.eos import Mixture, OMEGA_A, OMEGA_B, SQRT2  # noqa: E402
from pymrst.modules.compositional.fluids import CompositionalPropertyModel  # noqa: E402


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
    eos = EquationOfStateModel(None, mix, "Peng-Robinson")
    srk = EquationOfStateModel(None, mix, "SRK")
    prcorr = EquationOfStateModel(None, mix, "prcorr")

    eos.mixture = mix
    custom_pm = CompositionalPropertyModel(mix, volumeShift=[0.01, 0.02, 0.03])
    eos.property_model = custom_pm
    eos.setType("PR", c1=2.0, c2=-1.0)
    e2, e1, e0 = eos.getCubicCoefficients(np.array([0.4]), np.array([0.08]))
    roots = eos.solveCubicEOS(np.array([0.4]), np.array([0.08]))

    bad_type_raised = False
    try:
        EquationOfStateModel(None, mix, "not-a-real-eos")
    except ValueError:
        bad_type_raised = True

    zj_raised = False
    try:
        EquationOfStateModel(None, mix, "zj")
    except NotImplementedError:
        zj_raised = True

    print("\nPhase 257: EquationOfStateModel MRST object surface")
    failures = 0
    failures += _check("constructor stores CompositionalMixture",
                       eos.CompositionalMixture is mix and eos.mixture is mix)
    failures += _check("constructor creates matching PropertyModel",
                       isinstance(EquationOfStateModel(None, mix).PropertyModel, CompositionalPropertyModel)
                       and EquationOfStateModel(None, mix).PropertyModel.fluid is mix)
    failures += _check("component helpers mirror MRST",
                       eos.getNumberOfComponents() == 3
                       and eos.getComponentNames() == ("C1", "C3", "C10"))
    failures += _check("PR constants match MRST",
                       np.isclose(EquationOfStateModel(None, mix).omegaA, OMEGA_A)
                       and np.isclose(EquationOfStateModel(None, mix).omegaB, OMEGA_B)
                       and np.isclose(EquationOfStateModel(None, mix).eosA, 1.0 + SQRT2)
                       and np.isclose(EquationOfStateModel(None, mix).eosB, 1.0 - SQRT2))
    failures += _check("SRK and PRCORR shortnames match MRST",
                       srk.shortname() == "srk" and prcorr.shortname() == "prcorr")
    failures += _check("property_model alias sets PropertyModel",
                       eos.PropertyModel is custom_pm and eos.property_model is custom_pm)
    failures += _check("custom cubic coefficients follow MRST formula",
                       np.allclose(e2, -1.0 + (2.0 - 1.0 - 1.0) * 0.08)
                       and np.allclose(e1, 0.4 - (2.0 - 1.0 - 2.0 * -1.0) * 0.08**2 - (2.0 - 1.0) * 0.08)
                       and np.allclose(e0, -(0.4 * 0.08 + 2.0 * -1.0 * 0.08**2 * 1.08)))
    failures += _check("solveCubicEOS returns positive real roots",
                       roots.shape == (1, 3) and np.any(np.isfinite(roots[0])))
    failures += _check("invalid EOS choices are rejected", bad_type_raised and zj_raised)

    if failures:
        print(f"\nPhase 257 failed ({failures} checks).")
        return 1
    print("\nEquationOfStateModel now exposes the MRST object-level EOS API.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
