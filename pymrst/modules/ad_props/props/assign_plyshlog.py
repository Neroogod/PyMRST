"""Polymer shear thinning from PLYSHLOG (MRST ``props/assignPLYSHLOG.m``).

PLYSHLOG is the logarithmic form of PLYSHEAR: a viscosity multiplier against
shear rate or water velocity, stated on a log scale, with a reference
condition record alongside the table.

MRST stores the keyword whole and interprets it in the polymer model rather
than building closures here, so this file is one assignment. That is not an
oversight to correct: the abscissa's *meaning* depends on whether the deck
also states SHRATE - reciprocal time if it does, velocity if it does not - and
`convertPROPS` already applies the matching units. Building an interpolant
here would have to duplicate that decision.
"""

from typing import Any, Dict


def assign_plyshlog(f: Dict[str, Any], plyshlog, reg=None) -> Dict[str, Any]:
    """Store the PLYSHLOG record for the polymer model to interpret."""
    f["plyshlog"] = plyshlog
    return f


assignPLYSHLOG = assign_plyshlog


if __name__ == "__main__":
    import numpy as np

    RECORD = {"refcondition": np.array([[0.5, 0.0, 1.0]]),
              "data": [np.array([[-4.0, 1.0], [-2.0, 0.6], [0.0, 0.3]])]}

    def test_the_record_is_stored_whole() -> None:
        f = assign_plyshlog({}, RECORD)
        assert f["plyshlog"] is RECORD
        print("ok test_the_record_is_stored_whole")

    def test_no_closures_are_built() -> None:
        """The polymer model interprets the abscissa; see the module docstring."""
        f = assign_plyshlog({}, RECORD)
        assert not callable(f["plyshlog"])
        print("ok test_no_closures_are_built")

    def test_both_parts_of_the_record_survive() -> None:
        f = assign_plyshlog({}, RECORD)
        assert "refcondition" in f["plyshlog"] and "data" in f["plyshlog"]
        print("ok test_both_parts_of_the_record_survive")

    def test_existing_fields_are_preserved() -> None:
        f = assign_plyshlog({"muWMult": "kept"}, RECORD)
        assert f["muWMult"] == "kept"
        print("ok test_existing_fields_are_preserved")

    def test_it_does_not_collide_with_plyshear() -> None:
        from pymrst.modules.ad_props.props.assign_plyshear import assign_plyshear
        from pymrst.modules.ad_props.utils.handle_regions import Regions
        table = np.array([[0.0, 1.0], [1.0e-4, 0.5]])
        f = assign_plyshear({}, [table], Regions())
        f = assign_plyshlog(f, RECORD)
        assert "plyshearMult" in f and "plyshlog" in f
        print("ok test_it_does_not_collide_with_plyshear")

    test_the_record_is_stored_whole()
    test_no_closures_are_built()
    test_both_parts_of_the_record_survive()
    test_existing_fields_are_preserved()
    test_it_does_not_collide_with_plyshear()
    print("\nAll tests passed!")
