"""Wet-gas PVT from PVTG (MRST ``ad-props/props/assignPVTG.m``).

PVTG is PVTO's mirror image: gas carries vaporised oil, measured as Rv. But
the two tables are *keyed the other way round*, and that is the thing to get
right here.

- PVTO has one branch per Rs, with pressure along the branch.
- PVTG has one branch per **pressure**, with Rv along the branch.

So although both public functions take pressure first - ``bO(p, rs, flag)``
and ``bG(p, rv, flag)`` - the interpolator underneath receives them in
opposite orders: PVTO passes (pressure, Rs) and PVTG passes (Rv, pressure).
Swapping them does not raise; it silently interpolates along the wrong axis
and returns plausible numbers, so the order is asserted in the tests below
rather than left to reading.

``rvSat(p)`` is extended with (p=0, Rv=0) for the same reason PVTO's ``rsSat``
is: vaporised oil must vanish at surface conditions, and without the endpoint
a low-pressure cell can be handed a negative Rv.
"""

from typing import Any, Callable, Dict, List

import numpy as np

from pymrst.modules.ad_props.utils.interp_pvt import PVTTable, pvt_interp
from pymrst.modules.ad_props.utils.preprocess_table_pvt import (
    PackedPVT,
    preprocess_table_pvt,
)


def _pvt_table(packed, column: int) -> PVTTable:
    """A PVTTable of Rv against one property column of ``packed``.

    The saturated curve is taken from `preprocess_table_pvt`, not recomputed
    from each branch's first row. PVTG branches are written with Rv
    *decreasing* - saturated first, dry last - so preprocessing reverses them,
    and their first row is then the dry end. Reading the saturated point off
    the reversed branch gives Rv = 0 for every pressure, which collapses the
    saturation curve and makes b infinite.
    """
    data = np.asarray(packed.data, dtype=float)
    values = 1.0 / data[:, 1] if column == 1 else data[:, column]
    expanded = preprocess_table_pvt(PackedPVT(
        key=np.asarray(packed.key, dtype=float),
        pos=np.asarray(packed.pos, dtype=np.int64),
        data=np.column_stack([data[:, 0], values]),
    ))
    return PVTTable(
        key=expanded.key,
        branches=[(branch.x, branch.F[:, 0]) for branch in expanded.expanded],
        sat_x=expanded.sat.x,
        sat_f=expanded.sat.F[:, 0],
    )


def assign_pvtg(f: Dict[str, Any], pvtg, reg) -> Dict[str, Any]:
    """Store ``bG``, ``muG`` and ``rvSat``, one set per PVT region."""
    b_functions: List[Callable] = []
    mu_functions: List[Callable] = []
    rv_sat_functions: List[Callable] = []

    for index in range(reg.pvt):
        packed = pvtg[index]
        b_table = _pvt_table(packed, 1)          # column 1 is B, stored as 1/B
        mu_table = _pvt_table(packed, 2)

        p_vap = np.asarray(packed.key, dtype=float)
        pos = np.asarray(packed.pos, dtype=np.int64)
        rv = np.asarray(packed.data, dtype=float)[pos[:-1], 0]

        # (rv, pg), not (pg, rv): the branch abscissa comes first.
        method = reg.pvt_method_gas
        b_functions.append(
            lambda p, rv_, flag, *, t=b_table, m=method:
                pvt_interp(t, rv_, p, flag, m))
        mu_functions.append(
            lambda p, rv_, flag, *, t=mu_table, m=method:
                pvt_interp(t, rv_, p, flag, m))
        rv_sat_functions.append(
            lambda p, *, x=np.concatenate([[0.0], p_vap]),
            y=np.concatenate([[0.0], rv]): reg.interp(x, y, p))

    f["bG"], f["muG"], f["rvSat"] = b_functions, mu_functions, rv_sat_functions
    return f


assignPVTG = assign_pvtg


if __name__ == "__main__":
    from pymrst.modules.ad_props.utils.handle_regions import Regions

    def _packed():
        """Two pressure branches; Rv decreases along each, as decks write it."""
        return PackedPVT(
            key=np.array([1.0e7, 2.0e7]),       # the key is PRESSURE
            pos=np.array([0, 2, 4]),
            data=np.array([
                [1.0e-4, 1.0e-2, 1.4e-5],       # p = 1e7, saturated Rv
                [0.0,    1.1e-2, 1.5e-5],       #          dry gas
                [3.0e-4, 5.0e-3, 1.8e-5],       # p = 2e7, saturated Rv
                [0.0,    5.5e-3, 1.9e-5],       #          dry gas
            ]))

    def test_b_on_the_saturated_curve_is_the_tabulated_reciprocal() -> None:
        f = assign_pvtg({}, [_packed()], Regions())
        got = f["bG"][0](np.array([1.0e7]), np.array([1.0e-4]),
                         np.array([True]))
        assert np.isclose(got, 1.0 / 1.0e-2), got
        print("ok test_b_on_the_saturated_curve_is_the_tabulated_reciprocal")

    def test_viscosity_on_the_saturated_curve() -> None:
        f = assign_pvtg({}, [_packed()], Regions())
        got = f["muG"][0](np.array([2.0e7]), np.array([3.0e-4]),
                          np.array([True]))
        assert np.isclose(got, 1.8e-5), got
        print("ok test_viscosity_on_the_saturated_curve")

    def test_undersaturated_gas_follows_its_pressure_branch() -> None:
        """At fixed p, drier gas has a different b - that is the branch axis."""
        f = assign_pvtg({}, [_packed()], Regions())
        got = f["bG"][0](np.array([1.0e7]), np.array([0.0]), np.array([False]))
        assert np.isclose(got, 1.0 / 1.1e-2), got
        print("ok test_undersaturated_gas_follows_its_pressure_branch")

    def test_rv_sat_recovers_the_saturated_rv_at_each_pressure() -> None:
        f = assign_pvtg({}, [_packed()], Regions())
        assert np.isclose(f["rvSat"][0](1.0e7), 1.0e-4)
        assert np.isclose(f["rvSat"][0](2.0e7), 3.0e-4)
        print("ok test_rv_sat_recovers_the_saturated_rv_at_each_pressure")

    def test_rv_sat_vanishes_at_zero_pressure() -> None:
        f = assign_pvtg({}, [_packed()], Regions())
        assert np.isclose(f["rvSat"][0](0.0), 0.0)
        probe = np.linspace(0.0, 2.0e7, 50)
        assert np.all(f["rvSat"][0](probe) >= 0.0)
        print("ok test_rv_sat_vanishes_at_zero_pressure")

    def test_the_key_axis_is_pressure_not_rv() -> None:
        """The swap this file exists to get right.

        Holding Rv fixed and moving pressure must walk *between* branches. The
        expected value is MRST's, from `ref_ad_props_assign.mat`: 'linshift'
        shifts each branch's abscissa before interpolating, so the answer is
        not a plain blend of the two branches evaluated at Rv = 0 - that would
        give 136.36, and it is the wrong number.
        """
        f = assign_pvtg({}, [_packed()], Regions())
        got = f["bG"][0](np.array([1.5e7]), np.array([0.0]),
                         np.array([False]))
        assert np.isclose(got, 134.84848484848487), got
        naive = np.interp(1.5e7, [1.0e7, 2.0e7], [1 / 1.1e-2, 1 / 5.5e-3])
        assert not np.isclose(got, naive, rtol=1e-3), (got, naive)
        print("ok test_the_key_axis_is_pressure_not_rv")

    def test_swapping_the_arguments_would_give_a_different_answer() -> None:
        """A guard on the guard: the two orders are genuinely distinguishable."""
        f = assign_pvtg({}, [_packed()], Regions())
        correct = f["bG"][0](np.array([1.0e7]), np.array([0.0]),
                             np.array([False]))
        table = _pvt_table(_packed(), 1)
        swapped = pvt_interp(table, np.array([1.0e7]), np.array([0.0]),
                             np.array([False]))
        assert not np.isclose(correct, swapped), (correct, swapped)
        print("ok test_swapping_the_arguments_would_give_a_different_answer")

    def test_one_set_of_functions_per_pvt_region() -> None:
        f = assign_pvtg({}, [_packed(), _packed()], Regions(pvt=2))
        for name in ("bG", "muG", "rvSat"):
            assert len(f[name]) == 2, name
        print("ok test_one_set_of_functions_per_pvt_region")

    test_b_on_the_saturated_curve_is_the_tabulated_reciprocal()
    test_viscosity_on_the_saturated_curve()
    test_undersaturated_gas_follows_its_pressure_branch()
    test_rv_sat_recovers_the_saturated_rv_at_each_pressure()
    test_rv_sat_vanishes_at_zero_pressure()
    test_the_key_axis_is_pressure_not_rv()
    test_swapping_the_arguments_would_give_a_different_answer()
    test_one_set_of_functions_per_pvt_region()
    print("\nAll tests passed!")
