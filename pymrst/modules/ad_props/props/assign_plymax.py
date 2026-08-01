"""Maximum polymer concentration from PLYMAX (MRST ``props/assignPLYMAX.m``).

PLYMAX gives the polymer concentration used to normalise the Todd-Longstaff
mixing model. Only the first field is read; the second, the maximum salt
concentration, is not used by MRST.

It is a normalising constant, not a cap - a concentration above it is not
clipped, it simply extrapolates the mixing rule.
"""

from typing import Any, Dict

import numpy as np


def assign_plymax(f: Dict[str, Any], plymax, reg=None) -> Dict[str, Any]:
    """Store ``cpmax``, the polymer concentration used for mixing."""
    values = np.asarray(plymax, dtype=float).ravel()
    f["cpmax"] = values[0]
    return f


assignPLYMAX = assign_plymax


if __name__ == "__main__":

    def test_the_first_field_is_taken() -> None:
        f = assign_plymax({}, [3.0, 0.0])
        assert np.isclose(f["cpmax"], 3.0)
        print("ok test_the_first_field_is_taken")

    def test_the_second_field_is_ignored() -> None:
        assert assign_plymax({}, [3.0, 99.0])["cpmax"] == 3.0
        print("ok test_the_second_field_is_ignored")

    def test_a_bare_scalar_is_accepted() -> None:
        assert assign_plymax({}, 3.0)["cpmax"] == 3.0
        print("ok test_a_bare_scalar_is_accepted")

    def test_existing_fields_are_preserved() -> None:
        assert assign_plymax({"bW": "kept"}, [3.0])["bW"] == "kept"
        print("ok test_existing_fields_are_preserved")

    test_the_first_field_is_taken()
    test_the_second_field_is_ignored()
    test_a_bare_scalar_is_accepted()
    test_existing_fields_are_preserved()
    print("\nAll tests passed!")
