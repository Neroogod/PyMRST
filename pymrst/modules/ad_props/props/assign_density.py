"""Surface densities from DENSITY (MRST ``ad-props/props/assignDENSITY.m``).

DENSITY states one record per PVT region, in ECLIPSE's order: oil, water, gas.
The fluid stores them under the phase-named fields the rest of `ad-props`
reads, which is the only thing this function does - the reordering is the
whole content, and getting it wrong swaps the oil and water densities of every
deck that reaches it.
"""

from typing import Any, Dict

import numpy as np


def assign_density(f: Dict[str, Any], density, reg) -> Dict[str, Any]:
    """Store DENSITY's columns as ``rhoOS``, ``rhoWS`` and ``rhoGS``."""
    density = np.atleast_2d(np.asarray(density, dtype=float))
    f["rhoOS"] = density[:, 0]
    f["rhoWS"] = density[:, 1]
    f["rhoGS"] = density[:, 2]
    return f


assignDENSITY = assign_density


if __name__ == "__main__":
    from pymrst.modules.ad_props.utils.handle_regions import Regions

    def test_columns_map_to_the_named_phases() -> None:
        f = assign_density({}, [[800.0, 1000.0, 1.2]], Regions())
        assert f["rhoOS"] == [800.0] and f["rhoWS"] == [1000.0]
        assert f["rhoGS"] == [1.2]
        print("ok test_columns_map_to_the_named_phases")

    def test_one_entry_per_pvt_region() -> None:
        f = assign_density({}, [[800.0, 1000.0, 1.2],
                                [820.0, 1010.0, 1.3]], Regions(pvt=2))
        assert np.allclose(f["rhoOS"], [800.0, 820.0]), f["rhoOS"]
        print("ok test_one_entry_per_pvt_region")

    def test_a_single_record_is_accepted_flat() -> None:
        f = assign_density({}, np.array([800.0, 1000.0, 1.2]), Regions())
        assert np.allclose(f["rhoWS"], [1000.0])
        print("ok test_a_single_record_is_accepted_flat")

    def test_existing_fields_are_preserved() -> None:
        f = assign_density({"bO": "kept"}, [[800.0, 1000.0, 1.2]], Regions())
        assert f["bO"] == "kept"
        print("ok test_existing_fields_are_preserved")

    test_columns_map_to_the_named_phases()
    test_one_entry_per_pvt_region()
    test_a_single_record_is_accepted_flat()
    test_existing_fields_are_preserved()
    print("\nAll tests passed!")
