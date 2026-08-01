from __future__ import annotations

import numpy as np
import pytest

from pymrst.modules.compositional.fluids import TableCompositionalMixture
from pymrst.modules.compositional.utils import initDeckEOSModel
from pymrst.modules.compositional.utils.cool_prop_fluids_structs import (
    coolPropFluidsStructs,
)


def test_complete_mrst_coolprop_table_and_case_insensitive_lookup():
    rows = coolPropFluidsStructs()
    assert len(rows) == 122
    assert rows[0].name == "R507A"
    assert rows[-1].name == "R410A"
    mixture = TableCompositionalMixture(["methane", "N-DECANE"], ["C1", "C10"])
    assert mixture.names == ("C1", "C10")
    assert np.allclose(mixture.Tcrit, [190.564, 617.7])
    assert np.allclose(mixture.Pcrit, [4_599_200.0, 2_103_000.0])
    assert np.allclose(mixture.Vcrit, mixture.molarMass / [162.660002678, 233.3419552])


def test_table_mixture_rejects_unknown_fluid_and_bad_output_names():
    with pytest.raises(ValueError, match="NotAFluid"):
        TableCompositionalMixture("NotAFluid")
    with pytest.raises(AssertionError, match="same length"):
        TableCompositionalMixture(["Methane", "Ethane"], ["C1"])


def test_multiple_eos_regions_reproduce_mrst_region_one_fallback():
    deck = {
        "PROPS": {
            "CNAMES": ["N2", "CO2"],
            "TCRIT": [126.192, 304.1282],
            "PCRIT": [3.3958e6, 7.3773e6],
            "ZCRIT": [0.291, 0.274],
            "ACF": [0.0372, 0.22394],
            "MW": [0.02801348, 0.0440098],
            "EOS": ["SRK", "PR"],
        }
    }
    with pytest.warns(RuntimeWarning, match="using region 1"):
        model = initDeckEOSModel(deck)
    assert model.shortname() == "srk"
    assert np.isclose(model.omegaA, 0.427480)
    assert np.isclose(model.omegaB, 0.086640)

