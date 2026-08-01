"""Polymer rock properties from PLYROCK (MRST ``props/assignPLYROCK.m``).

Five per-region numbers describing how the rock treats polymer:

- ``dps``    dead pore space, the fraction of pore volume polymer cannot enter;
- ``rrf``    residual resistance factor, the permeability reduction polymer
             leaves behind after it has passed;
- ``rhoR``   rock mass density, which converts adsorption per unit mass to per
             unit volume;
- ``adsInx`` which adsorption model to use;
- ``adsMax`` the maximum adsorption.

Dead pore space is what makes polymer travel *faster* than the water carrying
it: excluded from part of the pore volume, it sees a smaller volume to fill.
Setting it to zero removes that effect entirely, so an unstated value being
read as zero rather than as absent changes the flood's arrival time.

These are values, not functions, and they expand to one entry per cell when
the deck has several saturation regions.
"""

from typing import Any, Dict

import numpy as np

# Column order of the PLYROCK record.
FIELDS = ("dps", "rrf", "rhoR", "adsInx", "adsMax")


def assign_plyrock(f: Dict[str, Any], plyrock, reg) -> Dict[str, Any]:
    """Store the five PLYROCK fields, per region or per cell."""
    plyrock = np.atleast_2d(np.asarray(plyrock, dtype=float))
    rows = 0 if reg.count("SAT") == 1 else reg.SATNUM
    for column, name in enumerate(FIELDS):
        f[name] = plyrock[rows, column]
    return f


assignPLYROCK = assign_plyrock


if __name__ == "__main__":
    from pymrst.modules.ad_props.utils.handle_regions import Regions

    RECORD = [0.05, 1.3, 2650.0, 1.0, 2.5e-5]

    def test_the_five_fields_are_named_in_column_order() -> None:
        f = assign_plyrock({}, [RECORD], Regions())
        assert np.isclose(f["dps"], 0.05)
        assert np.isclose(f["rrf"], 1.3)
        assert np.isclose(f["rhoR"], 2650.0)
        assert np.isclose(f["adsInx"], 1.0)
        assert np.isclose(f["adsMax"], 2.5e-5)
        print("ok test_the_five_fields_are_named_in_column_order")

    def test_the_values_are_not_functions() -> None:
        f = assign_plyrock({}, [RECORD], Regions())
        assert not callable(f["dps"])
        print("ok test_the_values_are_not_functions")

    def test_several_regions_expand_through_satnum() -> None:
        second = list(RECORD)
        second[0] = 0.10
        reg = Regions(SATNUM=np.array([0, 1, 1]),
                      SATINX=[np.array([0]), np.array([1, 2])])
        f = assign_plyrock({}, [RECORD, second], reg)
        assert np.allclose(f["dps"], [0.05, 0.10, 0.10]), f["dps"]
        print("ok test_several_regions_expand_through_satnum")

    def test_zero_dead_pore_space_is_a_real_setting() -> None:
        """It removes the polymer's velocity advantage rather than defaulting."""
        record = list(RECORD)
        record[0] = 0.0
        f = assign_plyrock({}, [record], Regions())
        assert f["dps"] == 0.0
        print("ok test_zero_dead_pore_space_is_a_real_setting")

    def test_existing_fields_are_preserved() -> None:
        f = assign_plyrock({"bW": "kept"}, [RECORD], Regions())
        assert f["bW"] == "kept"
        print("ok test_existing_fields_are_preserved")

    test_the_five_fields_are_named_in_column_order()
    test_the_values_are_not_functions()
    test_several_regions_expand_through_satnum()
    test_zero_dead_pore_space_is_a_real_setting()
    test_existing_fields_are_preserved()
    print("\nAll tests passed!")
