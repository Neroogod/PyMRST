"""Surfactant rock properties from SURFROCK (MRST ``props/assignSURFROCK.m``).

Two per-region numbers: ``adsInxSft``, which adsorption model the surfactant
uses, and ``rhoRSft``, the rock mass density that converts adsorption per unit
mass to per unit volume.

It partitions on the **surfactant** regions, not the saturation ones. A deck
whose surfactant partition differs from its saturation partition gets the
surfactant one here, which is the point of having a separate SURFNUM.
"""

from typing import Any, Dict

import numpy as np

FIELDS = ("adsInxSft", "rhoRSft")


def assign_surfrock(f: Dict[str, Any], surfrock, reg) -> Dict[str, Any]:
    """Store ``adsInxSft`` and ``rhoRSft``, per region or per cell."""
    surfrock = np.atleast_2d(np.asarray(surfrock, dtype=float))
    rows = 0 if reg.count("SURF") == 1 else reg.SURFNUM
    for column, name in enumerate(FIELDS):
        f[name] = surfrock[rows, column]
    return f


assignSURFROCK = assign_surfrock


if __name__ == "__main__":
    from pymrst.modules.ad_props.utils.handle_regions import Regions

    RECORD = [2.0, 2650.0]

    def test_the_two_fields_are_named_in_column_order() -> None:
        f = assign_surfrock({}, [RECORD], Regions())
        assert np.isclose(f["adsInxSft"], 2.0)
        assert np.isclose(f["rhoRSft"], 2650.0)
        print("ok test_the_two_fields_are_named_in_column_order")

    def test_it_partitions_on_surfnum_not_satnum() -> None:
        second = [3.0, 2400.0]
        reg = Regions(SATNUM=np.array([0, 0]), SATINX=[np.array([0, 1])],
                      SURFNUM=np.array([0, 1]),
                      SURFINX=[np.array([0]), np.array([1])])
        f = assign_surfrock({}, [RECORD, second], reg)
        assert np.allclose(f["rhoRSft"], [2650.0, 2400.0]), f["rhoRSft"]
        print("ok test_it_partitions_on_surfnum_not_satnum")

    def test_a_single_surfactant_region_gives_scalars() -> None:
        f = assign_surfrock({}, [RECORD], Regions())
        assert np.ndim(f["rhoRSft"]) == 0, f["rhoRSft"]
        print("ok test_a_single_surfactant_region_gives_scalars")

    def test_it_does_not_overwrite_the_polymer_rock_fields() -> None:
        from pymrst.modules.ad_props.props.assign_plyrock import assign_plyrock
        f = assign_plyrock({}, [[0.05, 1.3, 2650.0, 1.0, 2.5e-5]], Regions())
        f = assign_surfrock(f, [[2.0, 2400.0]], Regions())
        assert np.isclose(f["rhoR"], 2650.0), "polymer's own density stands"
        assert np.isclose(f["rhoRSft"], 2400.0)
        print("ok test_it_does_not_overwrite_the_polymer_rock_fields")

    test_the_two_fields_are_named_in_column_order()
    test_it_partitions_on_surfnum_not_satnum()
    test_a_single_surfactant_region_gives_scalars()
    test_it_does_not_overwrite_the_polymer_rock_fields()
    print("\nAll tests passed!")
