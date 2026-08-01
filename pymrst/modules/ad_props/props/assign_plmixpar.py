"""Polymer mixing parameter from PLMIXPAR (MRST ``props/assignPLMIXPAR.m``).

PLMIXPAR gives the Todd-Longstaff mixing parameter for polymer: 0 for no
mixing between polymer solution and water, 1 for full mixing.

It sets only ``mixPar``. TLMIXPAR sets the same field and additionally
``mixParRho``, so a deck stating both gets whichever is assigned last -
see `assign_tlmixpar`.
"""

from typing import Any, Dict

import numpy as np


def assign_plmixpar(f: Dict[str, Any], plmixpar, reg=None) -> Dict[str, Any]:
    """Store ``mixPar``, the Todd-Longstaff mixing parameter."""
    values = np.asarray(plmixpar, dtype=float).ravel()
    f["mixPar"] = values[0]
    return f


assignPLMIXPAR = assign_plmixpar


if __name__ == "__main__":

    def test_the_value_is_stored() -> None:
        assert np.isclose(assign_plmixpar({}, [1.0])["mixPar"], 1.0)
        print("ok test_the_value_is_stored")

    def test_it_does_not_set_the_density_mixing_parameter() -> None:
        """That is TLMIXPAR's, and conflating them changes mixture density."""
        assert "mixParRho" not in assign_plmixpar({}, [1.0])
        print("ok test_it_does_not_set_the_density_mixing_parameter")

    def test_zero_means_no_mixing() -> None:
        assert assign_plmixpar({}, [0.0])["mixPar"] == 0.0
        print("ok test_zero_means_no_mixing")

    test_the_value_is_stored()
    test_it_does_not_set_the_density_mixing_parameter()
    test_zero_means_no_mixing()
    print("\nAll tests passed!")
