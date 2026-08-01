"""Todd-Longstaff mixing parameters from TLMIXPAR (MRST ``props/assignTLMIXPAR.m``).

TLMIXPAR sets the mixing parameter for viscosity and, optionally, a separate
one for density. When only one value is given, density reuses it.

The relationship with PLMIXPAR is worth stating, because the two overlap:
PLMIXPAR sets ``mixPar`` alone, TLMIXPAR sets ``mixPar`` *and* ``mixParRho``.
A deck stating both ends up with whichever was assigned last, and MRST's
readPROPS notes that TLMIXPAR is the deprecated spelling - a polymer run
should use PLMIXPAR.

Separating the two parameters matters when they differ: viscosity mixing is
strongly non-linear in the parameter while density mixing is linear, so
forcing them equal changes the mixture density of any run that intended
otherwise.
"""

from typing import Any, Dict

import numpy as np


def assign_tlmixpar(f: Dict[str, Any], tlmixpar, reg=None) -> Dict[str, Any]:
    """Store ``mixPar`` and ``mixParRho``."""
    values = np.asarray(tlmixpar, dtype=float).ravel()
    f["mixPar"] = values[0]
    # A second field overrides the density parameter; otherwise it follows the
    # viscosity one.
    f["mixParRho"] = values[1] if values.size >= 2 and not np.isnan(values[1]) \
        else values[0]
    return f


assignTLMIXPAR = assign_tlmixpar


if __name__ == "__main__":

    def test_one_value_sets_both_parameters() -> None:
        f = assign_tlmixpar({}, [0.7])
        assert np.isclose(f["mixPar"], 0.7)
        assert np.isclose(f["mixParRho"], 0.7)
        print("ok test_one_value_sets_both_parameters")

    def test_a_second_value_overrides_the_density_parameter() -> None:
        f = assign_tlmixpar({}, [0.7, 0.3])
        assert np.isclose(f["mixPar"], 0.7)
        assert np.isclose(f["mixParRho"], 0.3)
        print("ok test_a_second_value_overrides_the_density_parameter")

    def test_the_two_parameters_stay_independent() -> None:
        f = assign_tlmixpar({}, [1.0, 0.0])
        assert f["mixPar"] != f["mixParRho"]
        print("ok test_the_two_parameters_stay_independent")

    def test_an_unstated_second_field_falls_back() -> None:
        """readPROPS defaults TLMIXPAR's fields to NaN, not to zero."""
        f = assign_tlmixpar({}, [0.7, np.nan])
        assert np.isclose(f["mixParRho"], 0.7), f["mixParRho"]
        print("ok test_an_unstated_second_field_falls_back")

    def test_it_sets_more_than_plmixpar_does() -> None:
        from pymrst.modules.ad_props.props.assign_plmixpar import assign_plmixpar
        assert "mixParRho" not in assign_plmixpar({}, [0.7])
        assert "mixParRho" in assign_tlmixpar({}, [0.7])
        print("ok test_it_sets_more_than_plmixpar_does")

    def test_a_bare_scalar_is_accepted() -> None:
        assert assign_tlmixpar({}, 0.5)["mixParRho"] == 0.5
        print("ok test_a_bare_scalar_is_accepted")

    test_one_value_sets_both_parameters()
    test_a_second_value_overrides_the_density_parameter()
    test_the_two_parameters_stay_independent()
    test_an_unstated_second_field_falls_back()
    test_it_sets_more_than_plmixpar_does()
    test_a_bare_scalar_is_accepted()
    print("\nAll tests passed!")
