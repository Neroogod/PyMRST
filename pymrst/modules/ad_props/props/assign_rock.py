"""Rock compressibility from ROCK (MRST ``ad-props/props/assignROCK.m``).

The pore-volume multiplier, as a second-order expansion around the reference
pressure - the same form PVTW uses for water:

    pvMult(p) = 1 + x + x^2/2,   x = c_R (p - p_ref)

With a single PVT region the fluid holds one function rather than a list. That
asymmetry is MRST's, and it is load-bearing: the single-region case is the
common one, and callers index `pvMultR` only when they know there are several.
"""

from typing import Any, Callable, Dict

import numpy as np


def pv_mult(p, c_rock, p_ref):
    """Pore-volume multiplier, ECLIPSE's second-order expansion."""
    x = c_rock * (p - p_ref)
    return 1.0 + x + 0.5 * x ** 2


def _tabulated(c_rock: float, p_ref: float, reg) -> Callable:
    """Evaluate directly, or through a table when ``reg.prange`` is set."""
    pressures = reg.prange
    if pressures is None or len(pressures) == 0:
        return lambda p: pv_mult(p, c_rock, p_ref)
    sampled = pv_mult(np.asarray(pressures, dtype=float), c_rock, p_ref)
    return lambda p: reg.interp_uniform(pressures, sampled, p)


def assign_rock(f: Dict[str, Any], rock, reg) -> Dict[str, Any]:
    """Store ``pvMultR``: one function, or one per PVT region."""
    rock = np.atleast_2d(np.asarray(rock, dtype=float))
    p_ref, c_rock = rock[:, 0], rock[:, 1]

    if reg.pvt == 1:
        f["pvMultR"] = _tabulated(c_rock[0], p_ref[0], reg)
    else:
        f["pvMultR"] = [_tabulated(c_rock[i], p_ref[i], reg)
                        for i in range(reg.pvt)]
    return f


assignROCK = assign_rock


if __name__ == "__main__":
    from pymrst.modules.ad_props.utils.handle_regions import Regions

    RECORD = [1.0132929e5, 4.3511321e-10]

    def test_the_multiplier_is_one_at_the_reference_pressure() -> None:
        f = assign_rock({}, [RECORD], Regions())
        assert np.isclose(f["pvMultR"](RECORD[0]), 1.0)
        print("ok test_the_multiplier_is_one_at_the_reference_pressure")

    def test_pore_volume_grows_with_pressure() -> None:
        f = assign_rock({}, [RECORD], Regions())
        assert f["pvMultR"](3.0e7) > 1.0 > f["pvMultR"](0.0)
        print("ok test_pore_volume_grows_with_pressure")

    def test_a_single_region_stores_a_bare_function() -> None:
        f = assign_rock({}, [RECORD], Regions())
        assert callable(f["pvMultR"]), "not a list when there is one region"
        print("ok test_a_single_region_stores_a_bare_function")

    def test_several_regions_store_a_list() -> None:
        f = assign_rock({}, [RECORD, [2.0e7, 1.0e-9]], Regions(pvt=2))
        assert isinstance(f["pvMultR"], list) and len(f["pvMultR"]) == 2
        assert np.isclose(f["pvMultR"][1](2.0e7), 1.0)
        print("ok test_several_regions_store_a_list")

    def test_the_expansion_matches_its_definition() -> None:
        c, p_ref, p = 1e-9, 1e7, 5e7
        x = c * (p - p_ref)
        f = assign_rock({}, [[p_ref, c]], Regions())
        assert np.isclose(f["pvMultR"](p), 1 + x + x ** 2 / 2)
        print("ok test_the_expansion_matches_its_definition")

    def test_prange_tabulates_without_changing_the_values() -> None:
        pressures = np.linspace(1e5, 1e8, 400)
        direct = assign_rock({}, [RECORD], Regions())["pvMultR"]
        tabulated = assign_rock({}, [RECORD],
                                Regions(prange=pressures))["pvMultR"]
        probe = np.array([1e6, 4e7, 9e7])
        assert np.allclose(direct(probe), tabulated(probe), rtol=1e-8)
        print("ok test_prange_tabulates_without_changing_the_values")

    test_the_multiplier_is_one_at_the_reference_pressure()
    test_pore_volume_grows_with_pressure()
    test_a_single_region_stores_a_bare_function()
    test_several_regions_store_a_list()
    test_the_expansion_matches_its_definition()
    test_prange_tabulates_without_changing_the_values()
    print("\nAll tests passed!")
