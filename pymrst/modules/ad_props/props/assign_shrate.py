"""Shear rate reference from SHRATE (MRST ``props/assignSHRATE.m``).

SHRATE gives the reference shear rate for the polymer shear-thinning
model. Its presence also changes how PLYSHLOG's abscissa is interpreted:
with SHRATE the table is in reciprocal time, without it in velocity.

`read_props` substitutes 4.8 for an unstated value, so a deck that names
the keyword without a value still gets ECLIPSE's default rather than zero.
"""

from typing import Any, Dict

import numpy as np


def assign_shrate(f: Dict[str, Any], shrate, reg=None) -> Dict[str, Any]:
    """Store ``shrate``, the reference shear rate."""
    values = np.asarray(shrate, dtype=float).ravel()
    f["shrate"] = values[0]
    return f


assignSHRATE = assign_shrate


if __name__ == "__main__":

    def test_the_value_is_stored() -> None:
        assert np.isclose(assign_shrate({}, [4.8])["shrate"], 4.8)
        print("ok test_the_value_is_stored")

    def test_a_bare_scalar_is_accepted() -> None:
        assert assign_shrate({}, 2.0)["shrate"] == 2.0
        print("ok test_a_bare_scalar_is_accepted")

    def test_the_reader_default_arrives_intact() -> None:
        """read_props substitutes 4.8 for NaN; nothing here overrides it."""
        assert assign_shrate({}, [4.8])["shrate"] == 4.8
        print("ok test_the_reader_default_arrives_intact")

    test_the_value_is_stored()
    test_a_bare_scalar_is_accepted()
    test_the_reader_default_arrives_intact()
    print("\nAll tests passed!")
