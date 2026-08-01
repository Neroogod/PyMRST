"""Live-oil PVT from PVTO (MRST ``ad-props/props/assignPVTO.m``).

PVTO is the black-oil model's most involved table. Oil holds dissolved gas, so
B and viscosity depend on pressure *and* on Rs, and the deck states them as one
pressure branch per Rs value. Each branch begins at its bubble point - where
the oil is saturated - and continues into the undersaturated region above it.

Four properties come out of it:

- ``bO`` and ``muO``, evaluated at (p, Rs) with a flag saying which cells are
  saturated. The flag is not redundant with comparing Rs against Rs_sat: near
  the bubble point the two disagree by rounding, and the caller knows which
  side it intends.
- ``rsSat(p)``, the saturation envelope - the maximum gas the oil can hold.
- ``pb(rs)``, its inverse, the bubble-point pressure.

Both envelopes get an extra endpoint, and each guards a different failure:

- `rsSat` is extended with (p=0, Rs=0), because Rs must vanish at surface
  conditions. Without it a low-pressure cell interpolates below the table's
  first bubble point and can be assigned a negative dissolved-gas content.
- `pb` is extended with (Rs=-1, p=p_bub[0]), so a slightly negative Rs - which
  a Newton step can produce - still returns the lowest bubble point rather
  than extrapolating below it.
"""

from typing import Any, Callable, Dict, List

import numpy as np

from pymrst.modules.ad_props.utils.interp_pvt import PVTTable, pvt_interp
from pymrst.modules.ad_props.utils.preprocess_table_pvt import (
    PackedPVT,
    preprocess_table_pvt,
)


def _pvt_table(packed, column: int) -> PVTTable:
    """A PVTTable of pressure against one property column of ``packed``.

    The saturated curve comes from `preprocess_table_pvt`, which reads it off
    the table in deck order. Recomputing it from each branch's first row would
    agree here - PVTO branches run upward in pressure - but not for PVTG,
    whose branches are reversed during preprocessing. Taking it from the same
    place in both keeps the two files honest about the same rule.
    """
    data = np.asarray(packed.data, dtype=float)
    values = 1.0 / data[:, 1] if column == 1 else data[:, column]
    expanded = preprocess_table_pvt(PackedPVT(
        key=np.asarray(packed.key, dtype=float),
        pos=np.asarray(packed.pos, dtype=np.int64),
        data=np.column_stack([data[:, 0], values]),
    ))
    # One property column, so hand the interpolator 1-D values: preprocess
    # keeps F two-dimensional to allow several columns at once.
    return PVTTable(
        key=expanded.key,
        branches=[(branch.x, branch.F[:, 0]) for branch in expanded.expanded],
        sat_x=expanded.sat.x,
        sat_f=expanded.sat.F[:, 0],
    )


def assign_pvto(f: Dict[str, Any], pvto, reg) -> Dict[str, Any]:
    """Store ``bO``, ``muO``, ``rsSat`` and ``pb``, one set per PVT region."""
    b_functions: List[Callable] = []
    mu_functions: List[Callable] = []
    rs_sat_functions: List[Callable] = []
    pb_functions: List[Callable] = []

    for index in range(reg.pvt):
        packed = pvto[index]
        b_table = _pvt_table(packed, 1)          # column 1 is B, stored as 1/B
        mu_table = _pvt_table(packed, 2)

        key = np.asarray(packed.key, dtype=float)
        pos = np.asarray(packed.pos, dtype=np.int64)
        p_bub = np.asarray(packed.data, dtype=float)[pos[:-1], 0]

        method = reg.pvt_method_oil
        b_functions.append(
            lambda p, rs, flag, *, t=b_table, m=method:
                pvt_interp(t, p, rs, flag, m))
        mu_functions.append(
            lambda p, rs, flag, *, t=mu_table, m=method:
                pvt_interp(t, p, rs, flag, m))
        rs_sat_functions.append(
            lambda p, *, x=np.concatenate([[0.0], p_bub]),
            y=np.concatenate([[0.0], key]): reg.interp(x, y, p))
        pb_functions.append(
            lambda rs, *, x=np.concatenate([[-1.0], key]),
            y=np.concatenate([p_bub[:1], p_bub]): reg.interp(x, y, rs))

    f["bO"], f["muO"] = b_functions, mu_functions
    f["rsSat"], f["pb"] = rs_sat_functions, pb_functions
    return f


assignPVTO = assign_pvto


if __name__ == "__main__":
    from pymrst.modules.ad_props.utils.handle_regions import Regions

    def _packed():
        """Two Rs branches, each with a bubble point and one undersaturated row."""
        return PackedPVT(
            key=np.array([10.0, 50.0]),
            pos=np.array([0, 2, 4]),
            data=np.array([
                [1.0e7, 1.20, 1.0e-3],      # Rs = 10, bubble point
                [2.0e7, 1.18, 1.1e-3],      #         undersaturated
                [2.0e7, 1.40, 0.8e-3],      # Rs = 50, bubble point
                [3.0e7, 1.37, 0.9e-3],      #         undersaturated
            ]))

    def test_b_on_the_saturated_curve_is_the_tabulated_reciprocal() -> None:
        f = assign_pvto({}, [_packed()], Regions())
        got = f["bO"][0](np.array([1.0e7]), np.array([10.0]), np.array([True]))
        assert np.isclose(got, 1.0 / 1.20), got
        print("ok test_b_on_the_saturated_curve_is_the_tabulated_reciprocal")

    def test_viscosity_on_the_saturated_curve() -> None:
        f = assign_pvto({}, [_packed()], Regions())
        got = f["muO"][0](np.array([2.0e7]), np.array([50.0]), np.array([True]))
        assert np.isclose(got, 0.8e-3), got
        print("ok test_viscosity_on_the_saturated_curve")

    def test_undersaturated_oil_follows_its_own_branch() -> None:
        f = assign_pvto({}, [_packed()], Regions())
        got = f["bO"][0](np.array([2.0e7]), np.array([10.0]), np.array([False]))
        assert np.isclose(got, 1.0 / 1.18), got
        print("ok test_undersaturated_oil_follows_its_own_branch")

    def test_rs_sat_recovers_the_bubble_points() -> None:
        f = assign_pvto({}, [_packed()], Regions())
        assert np.isclose(f["rsSat"][0](1.0e7), 10.0)
        assert np.isclose(f["rsSat"][0](2.0e7), 50.0)
        print("ok test_rs_sat_recovers_the_bubble_points")

    def test_rs_sat_vanishes_at_zero_pressure() -> None:
        """The added endpoint: dissolved gas must go to zero at the surface."""
        f = assign_pvto({}, [_packed()], Regions())
        assert np.isclose(f["rsSat"][0](0.0), 0.0)
        assert f["rsSat"][0](5.0e6) > 0.0, "and stays positive in between"
        print("ok test_rs_sat_vanishes_at_zero_pressure")

    def test_rs_sat_never_goes_negative_below_the_first_bubble_point() -> None:
        f = assign_pvto({}, [_packed()], Regions())
        probe = np.linspace(0.0, 1.0e7, 50)
        assert np.all(f["rsSat"][0](probe) >= 0.0)
        print("ok test_rs_sat_never_goes_negative_below_the_first_bubble_point")

    def test_pb_inverts_rs_sat() -> None:
        f = assign_pvto({}, [_packed()], Regions())
        assert np.isclose(f["pb"][0](10.0), 1.0e7)
        assert np.isclose(f["pb"][0](50.0), 2.0e7)
        print("ok test_pb_inverts_rs_sat")

    def test_pb_is_flat_for_a_slightly_negative_rs() -> None:
        """A Newton step can propose rs < 0; pb must not extrapolate below."""
        f = assign_pvto({}, [_packed()], Regions())
        assert np.isclose(f["pb"][0](-0.5), 1.0e7)
        assert np.isclose(f["pb"][0](0.0), 1.0e7)
        print("ok test_pb_is_flat_for_a_slightly_negative_rs")

    def test_one_set_of_functions_per_pvt_region() -> None:
        f = assign_pvto({}, [_packed(), _packed()], Regions(pvt=2))
        for name in ("bO", "muO", "rsSat", "pb"):
            assert len(f[name]) == 2, name
        print("ok test_one_set_of_functions_per_pvt_region")

    def test_b_interpolates_the_reciprocal_not_the_fvf() -> None:
        f = assign_pvto({}, [_packed()], Regions())
        got = f["bO"][0](np.array([1.5e7]), np.array([10.0]), np.array([False]))
        reciprocal = np.interp(1.5e7, [1.0e7, 2.0e7], [1 / 1.20, 1 / 1.18])
        inverted = 1.0 / np.interp(1.5e7, [1.0e7, 2.0e7], [1.20, 1.18])
        assert np.isclose(got, reciprocal), (got, reciprocal)
        assert not np.isclose(got, inverted, rtol=1e-8)
        print("ok test_b_interpolates_the_reciprocal_not_the_fvf")

    test_b_on_the_saturated_curve_is_the_tabulated_reciprocal()
    test_viscosity_on_the_saturated_curve()
    test_undersaturated_oil_follows_its_own_branch()
    test_rs_sat_recovers_the_bubble_points()
    test_rs_sat_vanishes_at_zero_pressure()
    test_rs_sat_never_goes_negative_below_the_first_bubble_point()
    test_pb_inverts_rs_sat()
    test_pb_is_flat_for_a_slightly_negative_rs()
    test_one_set_of_functions_per_pvt_region()
    test_b_interpolates_the_reciprocal_not_the_fvf()
    print("\nAll tests passed!")
