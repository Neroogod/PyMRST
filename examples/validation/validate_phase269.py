"""Phase 269: initDeckEOSModel utility parity."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pymrst.modules.compositional.eos import R_GAS  # noqa: E402
from pymrst.modules.compositional.models.eos import EquationOfStateModel  # noqa: E402
from pymrst.modules.compositional.utils import initDeckEOSModel, init_deck_eos_model, list_utils  # noqa: E402


def _check(name: str, condition: bool) -> int:
    print(f"  {'ok  ' if condition else 'FAIL'} {name}")
    return 0 if condition else 1


def _deck_vcrit():
    return {
        "PROPS": {
            "CNAMES": ["C1", "C3", "C10"],
            "TCRIT": [190.56, 369.83, 617.7],
            "PCRIT": [4.5992e6, 4.248e6, 2.11e6],
            "VCRIT": [9.86e-5, 2.00e-4, 6.17e-4],
            "ACF": [0.0114, 0.1523, 0.4884],
            "MW": [0.01604, 0.04410, 0.14229],
            "BIC": [[0.0, 0.01, 0.02], [0.01, 0.0, 0.03], [0.02, 0.03, 0.0]],
            "EOS": "PR",
            "PRCORR": True,
            "OMEGAA": [[np.nan, 0.5, 0.5]],
            "OMEGAB": [[np.nan, 0.09, 0.09]],
            "SSHIFT": [0.01, 0.02, 0.03],
        }
    }


def _deck_zcrit_object():
    props = SimpleNamespace(
        CNAMES=["N2", "CO2"],
        TCRIT=np.array([126.192, 304.1282]),
        PCRIT=np.array([3.3958e6, 7.3773e6]),
        ZCRIT=np.array([0.291, 0.274]),
        ACF=np.array([0.0372, 0.22394]),
        MW=np.array([0.02801348, 0.0440098]),
        EOS=["SRK", "PR"],
        OMEGAA=np.array([[np.nan, np.nan]]),
        OMEGAB=np.array([[np.nan, np.nan]]),
    )
    return SimpleNamespace(PROPS=props)


def main() -> int:
    prcorr = initDeckEOSModel(_deck_vcrit())
    srk = init_deck_eos_model(_deck_zcrit_object())
    catalog = list_utils()

    expected_vcrit = R_GAS * np.array([0.291, 0.274]) * np.array([126.192, 304.1282]) / np.array([3.3958e6, 7.3773e6])

    print("\nPhase 269: initDeckEOSModel utility parity")
    failures = 0
    failures += _check("constructor returns EquationOfStateModel",
                       isinstance(prcorr, EquationOfStateModel)
                       and isinstance(srk, EquationOfStateModel))
    failures += _check("deck CNAMES and critical properties populate Mixture",
                       prcorr.CompositionalMixture.names == ("C1", "C3", "C10")
                       and np.allclose(prcorr.CompositionalMixture.Tcrit, [190.56, 369.83, 617.7])
                       and np.allclose(prcorr.CompositionalMixture.Pcrit, [4.5992e6, 4.248e6, 2.11e6])
                       and np.allclose(prcorr.CompositionalMixture.Vcrit, [9.86e-5, 2.00e-4, 6.17e-4]))
    failures += _check("BIC and volume shift are transferred",
                       np.allclose(prcorr.CompositionalMixture.getBinaryInteraction(),
                                   [[0.0, 0.01, 0.02], [0.01, 0.0, 0.03], [0.02, 0.03, 0.0]])
                       and np.allclose(prcorr.PropertyModel.volumeShift, [0.01, 0.02, 0.03]))
    failures += _check("PRCORR overrides PR deck EOS and omega defaults are filled",
                       prcorr.shortname() == "prcorr"
                       and np.isclose(prcorr.omegaA, 0.457235529)
                       and np.isclose(prcorr.omegaB, 0.077796074))
    failures += _check("object-style deck supports ZCRIT-derived VCRIT",
                       srk.CompositionalMixture.names == ("N2", "CO2")
                       and np.allclose(srk.CompositionalMixture.Vcrit, expected_vcrit))
    failures += _check("multiple EOS regions use region one with SRK constants",
                       srk.shortname() == "srk"
                       and np.isclose(srk.omegaA, 0.4274802)
                       and np.isclose(srk.omegaB, 0.08664035))
    failures += _check("reference p/T markers match MRST setup",
                       np.isclose(prcorr.reference_temperature, 288.15)
                       and np.isclose(prcorr.reference_pressure, 101325.0))
    failures += _check("utility catalog marks initDeckEOSModel implemented",
                       catalog["initDeckEOSModel"]["status"] == "implemented"
                       and catalog["initDeckEOSModel"]["python_module"] == "init_deck_eos_model")

    if failures:
        print(f"\nPhase 269 failed ({failures} checks).")
        return 1
    print("\ninitDeckEOSModel now builds EquationOfStateModel objects from parsed deck PROPS.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
