"""Validate ``fieldModelNorneExample`` against archived OPM field results."""

import numpy as np

from pymrst.modules.ad_blackoil.examples.field_model_norne_example import (
    setup_field_model_norne,
)


def validate():
    case = setup_field_model_norne()
    signature = case.regression_signature()
    np.testing.assert_array_equal(signature["dimensions"], [44431, 2, 36, 247, 65])
    np.testing.assert_allclose(signature["states"][:, 1], [
        27592890.01332968, 27257607.128585216, 30324153.52905683,
    ], rtol=2e-12)
    np.testing.assert_allclose(signature["time"], [0, 1548, 3312])
    assert len(case.deck.faults) == 977
    assert len(case.deck.fault_multipliers) == 60
    print("Norne full-field OPM import and scalable regression passed")


if __name__ == "__main__":
    validate()
