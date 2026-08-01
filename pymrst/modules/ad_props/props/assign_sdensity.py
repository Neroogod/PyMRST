"""Solvent surface density from SDENSITY (MRST ``props/assignSDENSITY.m``).

One value per PVT region. With a single region the scalar is stored directly;
with several it is expanded to one value per cell through PVTNUM, so callers
never have to know which form they were given.
"""

from typing import Any, Dict

import numpy as np


def assign_sdensity(f: Dict[str, Any], sdensity, reg) -> Dict[str, Any]:
    """Store SDENSITY as ``rhoSS``, per region or per cell."""
    sdensity = np.atleast_2d(np.asarray(sdensity, dtype=float))
    if reg.count("PVT") == 1:
        f["rhoSS"] = sdensity[0, 0]
    else:
        f["rhoSS"] = sdensity[reg.PVTNUM, 0]
    return f


assignSDENSITY = assign_sdensity


if __name__ == "__main__":
    from pymrst.modules.ad_props.utils.handle_regions import Regions

    def test_a_single_region_stores_the_scalar() -> None:
        f = assign_sdensity({}, [[700.0]], Regions())
        assert f["rhoSS"] == 700.0, f
        print("ok test_a_single_region_stores_the_scalar")

    def test_several_regions_expand_through_pvtnum() -> None:
        reg = Regions(PVTNUM=np.array([0, 1, 1, 0]),
                      PVTINX=[np.array([0, 3]), np.array([1, 2])])
        f = assign_sdensity({}, [[700.0], [750.0]], reg)
        assert np.allclose(f["rhoSS"], [700.0, 750.0, 750.0, 700.0]), f["rhoSS"]
        print("ok test_several_regions_expand_through_pvtnum")

    test_a_single_region_stores_the_scalar()
    test_several_regions_expand_through_pvtnum()
    print("\nAll tests passed!")
