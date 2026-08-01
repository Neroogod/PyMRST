"""Surface densities from GRAVITY (MRST ``props/assignGRAVITY.m``).

GRAVITY is DENSITY expressed the way the industry quotes it: oil as API
gravity, water and gas as specific gravities. The conversions are fixed
constants, so this ends where `assignDENSITY` does - the same three fields -
by a different route.

Oil follows the API definition, ``rho = 141.5 / (API + 131.5)`` relative to
water. Water and gas are simple multiples of their reference fluid, and note
the references differ: water's is water, gas's is *air*.
"""

from typing import Any, Dict

import numpy as np

from pymrst.utils.units.units import KILOGRAM, METER

# MRST's reference_density(): fresh water and air at standard conditions.
RHO_WATER = 1000.0 * KILOGRAM / METER ** 3
RHO_AIR = 1.22 * KILOGRAM / METER ** 3


def oil_density(rho_water: float, api) -> np.ndarray:
    """Oil density from API gravity."""
    return rho_water * (141.5 / (np.asarray(api, dtype=float) + 131.5))


def assign_gravity(f: Dict[str, Any], api, reg) -> Dict[str, Any]:
    """Store GRAVITY as ``rhoOS``, ``rhoWS`` and ``rhoGS``."""
    api = np.atleast_2d(np.asarray(api, dtype=float))
    rho_o = oil_density(RHO_WATER, api[:, 0])
    rho_w = RHO_WATER * api[:, 1]
    rho_g = RHO_AIR * api[:, 2]

    if reg.count("PVT") == 1:
        f["rhoOS"], f["rhoWS"], f["rhoGS"] = rho_o[0], rho_w[0], rho_g[0]
    else:
        f["rhoOS"] = rho_o[reg.PVTNUM]
        f["rhoWS"] = rho_w[reg.PVTNUM]
        f["rhoGS"] = rho_g[reg.PVTNUM]
    return f


assignGRAVITY = assign_gravity


if __name__ == "__main__":
    from pymrst.modules.ad_props.utils.handle_regions import Regions

    def test_ten_api_is_the_density_of_water() -> None:
        """The API scale is defined so that 10 API is exactly water."""
        assert np.isclose(oil_density(RHO_WATER, 10.0), RHO_WATER)
        print("ok test_ten_api_is_the_density_of_water")

    def test_heavier_oil_has_lower_api() -> None:
        assert oil_density(RHO_WATER, 20.0) > oil_density(RHO_WATER, 40.0)
        print("ok test_heavier_oil_has_lower_api")

    def test_the_defaults_reproduce_mrsts_template() -> None:
        """readPROPS defaults GRAVITY to 45.5 API, sg 1.0, gas sg 0.7773."""
        f = assign_gravity({}, [[45.5, 1.0, 0.7773]], Regions())
        assert np.isclose(f["rhoWS"], 1000.0), f["rhoWS"]
        assert np.isclose(f["rhoGS"], 1.22 * 0.7773), f["rhoGS"]
        assert np.isclose(f["rhoOS"], 1000.0 * 141.5 / (45.5 + 131.5))
        print("ok test_the_defaults_reproduce_mrsts_template")

    def test_gas_is_referenced_to_air_not_water() -> None:
        f = assign_gravity({}, [[30.0, 1.0, 1.0]], Regions())
        assert np.isclose(f["rhoGS"], RHO_AIR), f["rhoGS"]
        print("ok test_gas_is_referenced_to_air_not_water")

    def test_several_regions_expand_through_pvtnum() -> None:
        reg = Regions(PVTNUM=np.array([0, 1]),
                      PVTINX=[np.array([0]), np.array([1])])
        f = assign_gravity({}, [[30.0, 1.0, 0.8], [40.0, 1.05, 0.7]], reg)
        assert f["rhoOS"].shape == (2,), f["rhoOS"]
        assert f["rhoOS"][0] > f["rhoOS"][1], "30 API is heavier than 40 API"
        print("ok test_several_regions_expand_through_pvtnum")

    test_ten_api_is_the_density_of_water()
    test_heavier_oil_has_lower_api()
    test_the_defaults_reproduce_mrsts_template()
    test_gas_is_referenced_to_air_not_water()
    test_several_regions_expand_through_pvtnum()
    print("\nAll tests passed!")
