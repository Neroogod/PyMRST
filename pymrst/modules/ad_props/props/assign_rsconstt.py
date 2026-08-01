"""Constant dissolved gas from RSCONSTT (MRST ``props/assignRSCONSTT.m``).

RSCONSTT declares a fixed solution gas-oil ratio and its bubble-point pressure
per PVT region, for a run where Rs does not vary with pressure.

Unlike every other PVT keyword these are *values*, not functions: ``f['rs']``
and ``f['pBub']`` are numbers. Code that treats every PVT field as callable
will fail on a deck that uses this keyword.

MRST's readPROPS defaults both fields to -1 rather than 0, so an unstated
entry is recognisably absent instead of looking like "no dissolved gas".
"""

from typing import Any, Dict

import numpy as np


def assign_rsconstt(f: Dict[str, Any], rsconstt, reg) -> Dict[str, Any]:
    """Store ``rs`` and ``pBub``, per region or per cell."""
    rsconstt = np.atleast_2d(np.asarray(rsconstt, dtype=float))
    if reg.count("PVT") == 1:
        f["rs"] = rsconstt[0, 0]
        f["pBub"] = rsconstt[0, 1]
    else:
        f["rs"] = rsconstt[reg.PVTNUM, 0]
        f["pBub"] = rsconstt[reg.PVTNUM, 1]
    return f


assignRSCONSTT = assign_rsconstt


if __name__ == "__main__":
    from pymrst.modules.ad_props.utils.handle_regions import Regions

    def test_a_single_region_stores_scalars() -> None:
        f = assign_rsconstt({}, [[25.0, 1.5e7]], Regions())
        assert np.isclose(f["rs"], 25.0) and np.isclose(f["pBub"], 1.5e7)
        print("ok test_a_single_region_stores_scalars")

    def test_the_values_are_not_functions() -> None:
        """Every other PVT field is callable; these two are not."""
        f = assign_rsconstt({}, [[25.0, 1.5e7]], Regions())
        assert not callable(f["rs"]) and not callable(f["pBub"])
        print("ok test_the_values_are_not_functions")

    def test_several_regions_expand_through_pvtnum() -> None:
        reg = Regions(PVTNUM=np.array([0, 1, 1]),
                      PVTINX=[np.array([0]), np.array([1, 2])])
        f = assign_rsconstt({}, [[25.0, 1.5e7], [40.0, 2.5e7]], reg)
        assert np.allclose(f["rs"], [25.0, 40.0, 40.0]), f["rs"]
        assert np.allclose(f["pBub"], [1.5e7, 2.5e7, 2.5e7]), f["pBub"]
        print("ok test_several_regions_expand_through_pvtnum")

    def test_the_unstated_default_is_carried_through() -> None:
        """readPROPS defaults RSCONSTT to -1, meaning 'absent', not 'zero'."""
        f = assign_rsconstt({}, [[-1.0, -1.0]], Regions())
        assert f["rs"] == -1.0, f["rs"]
        print("ok test_the_unstated_default_is_carried_through")

    def test_existing_fields_are_preserved() -> None:
        f = assign_rsconstt({"bO": "kept"}, [[25.0, 1.5e7]], Regions())
        assert f["bO"] == "kept"
        print("ok test_existing_fields_are_preserved")

    test_a_single_region_stores_scalars()
    test_the_values_are_not_functions()
    test_several_regions_expand_through_pvtnum()
    test_the_unstated_default_is_carried_through()
    test_existing_fields_are_preserved()
    print("\nAll tests passed!")
