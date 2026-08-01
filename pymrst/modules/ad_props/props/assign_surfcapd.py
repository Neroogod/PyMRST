"""Capillary de-saturation from SURFCAPD (MRST ``props/assignSURFCAPD.m``).

SURFCAPD gives a miscibility factor against the logarithm of the capillary
number. It interpolates between the immiscible relative permeability curves
and the miscible straight lines, so it is how a surfactant flood's reduced
interfacial tension actually reaches the flow equations.

The abscissa is already a logarithm, so the value to look up is log10(Nc) and
not Nc. Taking the log twice, or not at all, moves the answer by orders of
magnitude without any error.

The field is named ``miscfact``, and it partitions on the **PVT** regions -
not the saturation ones, despite being a saturation-side quantity. That is
MRST's choice; a deck whose PVT and saturation partitions differ gets the PVT
one here.
"""

from typing import Any, Callable, Dict, List

import numpy as np

from pymrst.modules.ad_props.utils.extend_tab import extend_tab


def assign_surfcapd(f: Dict[str, Any], surfcapd, reg) -> Dict[str, Any]:
    """Store ``miscfact`` as one interpolant per PVT region."""
    curves: List[Callable] = []
    for index in range(reg.pvt):
        table = extend_tab(np.asarray(surfcapd[index], dtype=float))
        curves.append(
            lambda nc, *, a=table[:, 0], b=table[:, 1]: reg.interp(a, b, nc))
    f["miscfact"] = curves
    return f


assignSURFCAPD = assign_surfcapd


if __name__ == "__main__":
    from pymrst.modules.ad_props.utils.handle_regions import Regions

    #        log10(Nc)  miscibility
    TABLE = np.array([[-10.0, 0.0],
                      [-5.0, 0.5],
                      [-2.0, 1.0]])

    def test_the_factor_is_read_from_the_table() -> None:
        f = assign_surfcapd({}, [TABLE], Regions())
        assert np.isclose(f["miscfact"][0](-5.0), 0.5)
        print("ok test_the_factor_is_read_from_the_table")

    def test_the_abscissa_is_already_a_logarithm() -> None:
        """Looking up Nc rather than log10(Nc) lands off the end of the table."""
        f = assign_surfcapd({}, [TABLE], Regions())
        capillary_number = 1e-5
        assert np.isclose(f["miscfact"][0](np.log10(capillary_number)), 0.5)
        assert np.isclose(f["miscfact"][0](capillary_number), 1.0), \
            "the raw number saturates at the miscible end"
        print("ok test_the_abscissa_is_already_a_logarithm")

    def test_a_low_capillary_number_is_immiscible() -> None:
        f = assign_surfcapd({}, [TABLE], Regions())
        assert np.isclose(f["miscfact"][0](-20.0), 0.0)
        print("ok test_a_low_capillary_number_is_immiscible")

    def test_a_high_capillary_number_is_fully_miscible() -> None:
        f = assign_surfcapd({}, [TABLE], Regions())
        assert np.isclose(f["miscfact"][0](0.0), 1.0)
        print("ok test_a_high_capillary_number_is_fully_miscible")

    def test_it_partitions_on_pvt_regions() -> None:
        second = TABLE.copy()
        second[:, 1] *= 0.5
        f = assign_surfcapd({}, [TABLE, second], Regions(pvt=2, sat=1))
        assert len(f["miscfact"]) == 2
        assert np.isclose(f["miscfact"][1](-5.0), 0.25)
        print("ok test_it_partitions_on_pvt_regions")

    test_the_factor_is_read_from_the_table()
    test_the_abscissa_is_already_a_logarithm()
    test_a_low_capillary_number_is_immiscible()
    test_a_high_capillary_number_is_fully_miscible()
    test_it_partitions_on_pvt_regions()
    print("\nAll tests passed!")
