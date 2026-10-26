"""Regional PVT interpolation (MRST ``ad-props/utils/interpRegPVT.m``).

The entry point a black-oil property actually calls. For each region it splits
the cells by the saturation flag and sends each half to the right place:

- saturated cells follow the bubble-point locus, a one-dimensional lookup in
  pressure through `interp_reg`;
- undersaturated cells sit on a constant-Rs branch, a two-dimensional lookup
  through `interp_2d_pvt`.

That split is the whole content of the function. Sending a cell to the wrong
side produces a property that is smooth, plausible, and wrong only near the
bubble point - which is exactly where a black-oil solve spends its iterations.
"""

from typing import Sequence

import numpy as np

from pymrst.modules.ad_props.utils.get_reg_map import ALL
from pymrst.modules.ad_props.utils.interp_2d_pvt import interp_2d_pvt
from pymrst.modules.ad_props.utils.interp_reg import interp_reg
from pymrst.modules.ad_props.utils.preprocess_table_pvt import (
    PackedPVT,
    preprocess_table_pvt,
)
from pymrst.utils.adi import ADI, _diag_mul, ad_from_blocks, is_ad, value


def _saturated_table(table: PackedPVT, column: int) -> np.ndarray:
    if getattr(table, "sat", None) is None:
        table = preprocess_table_pvt(table)
    return np.column_stack([table.sat.x, table.sat.F[:, column]])


def _region_mask(reg_inx, region: int, size: int) -> np.ndarray:
    subs = reg_inx[region]
    if subs is ALL or isinstance(subs, slice):
        return np.ones(size, dtype=bool)
    mask = np.zeros(size, dtype=bool)
    mask[np.asarray(subs, dtype=np.int64)] = True
    return mask


def interp_reg_pvt(
    tables: Sequence[PackedPVT],
    x,
    v,
    flag,
    reg_inx: Sequence,
    *,
    column: int = 0,
):
    """Interpolate a PVT property at ``(x, v)``, per region and per regime.

    ``flag`` is True where the cell is saturated. Returns an ADI value when
    either ``x`` or ``v`` carries derivatives.
    """
    xi = np.atleast_1d(value(x)).astype(float)
    vi = np.atleast_1d(value(v)).astype(float)
    saturated = np.atleast_1d(np.asarray(flag, dtype=bool))

    values = np.zeros(xi.shape)
    dx = np.zeros(xi.shape)
    dv = np.zeros(xi.shape)

    for region in range(len(reg_inx)):
        in_region = _region_mask(reg_inx, region, xi.size)
        table = tables[region]

        on_sat = np.flatnonzero(saturated & in_region)
        if on_sat.size:
            sat_table = _saturated_table(table, column)
            values[on_sat], dx[on_sat] = interp_reg(
                [sat_table], xi[on_sat], [ALL], derivative=True)
            # along the saturated locus the property follows pressure alone
            dv[on_sat] = 0.0

        off_sat = np.flatnonzero((~saturated) & in_region)
        if off_sat.size:
            sub = interp_2d_pvt([table], xi[off_sat], vi[off_sat], [ALL],
                                column=column)
            values[off_sat] = sub
            from pymrst.modules.ad_props.utils.interp_2d_pvt import _kernel
            from pymrst.modules.ad_props.utils.interp_2d_pvt import _as_expanded
            _, dxs, dvs = _kernel(_as_expanded(table), xi[off_sat],
                                  vi[off_sat], column)
            dx[off_sat], dv[off_sat] = dxs, dvs

    if not is_ad(x) and not is_ad(v):
        return values
    jacobians = None
    if is_ad(x):
        jacobians = [_diag_mul(dx, j) for j in x.jac]
    if is_ad(v):
        other = [_diag_mul(dv, j) for j in v.jac]
        jacobians = other if jacobians is None else [
            a + b for a, b in zip(jacobians, other)]
    return ad_from_blocks(values, jacobians)


interpRegPVT = interp_reg_pvt


if __name__ == "__main__":

    def _table():
        """Branch key k has F = x + 10*k; the saturated locus is F = x."""
        data = np.array([
            [1.0, 1.0], [3.0, 3.0],            # key 0 saturated at x=1
            [2.0, 12.0], [3.0, 13.0],          # key 1 saturated at x=2
        ])
        return preprocess_table_pvt(
            PackedPVT(key=np.array([0.0, 1.0]), pos=np.array([0, 2, 4]),
                      data=data))

    def test_saturated_cells_follow_the_bubble_point_locus() -> None:
        tab = _table()
        out = interp_reg_pvt([tab], np.array([1.0, 2.0]), np.array([0.0, 1.0]),
                             np.array([True, True]), [ALL])
        assert np.allclose(out, [1.0, 12.0]), out
        print("ok test_saturated_cells_follow_the_bubble_point_locus")

    def test_undersaturated_cells_use_the_two_dimensional_branch() -> None:
        tab = _table()
        out = interp_reg_pvt([tab], np.array([3.0]), np.array([0.5]),
                             np.array([False]), [ALL])
        assert np.allclose(out, [8.0]), out     # halfway between 3 and 13
        print("ok test_undersaturated_cells_use_the_two_dimensional_branch")

    def test_the_flag_selects_between_the_two_regimes() -> None:
        """Same query, different flag, must give different answers."""
        tab = _table()
        query = (np.array([3.0]), np.array([0.5]))
        sat = interp_reg_pvt([tab], *query, np.array([True]), [ALL])
        und = interp_reg_pvt([tab], *query, np.array([False]), [ALL])
        assert not np.allclose(sat, und), (sat, und)
        print("ok test_the_flag_selects_between_the_two_regimes")

    def test_mixed_flags_are_handled_cell_by_cell() -> None:
        tab = _table()
        out = interp_reg_pvt([tab], np.array([3.0, 3.0]), np.array([0.5, 0.5]),
                             np.array([True, False]), [ALL])
        assert not np.isclose(out[0], out[1]), out
        assert np.allclose(out[1], 8.0), out
        print("ok test_mixed_flags_are_handled_cell_by_cell")

    def test_saturated_property_does_not_depend_on_the_key() -> None:
        from pymrst.utils.adi import init_variables_adi
        tab = _table()
        p, rs = init_variables_adi(np.array([2.0]), np.array([1.0]))
        out = interp_reg_pvt([tab], p, rs, np.array([True]), [ALL])
        assert isinstance(out, ADI)
        assert np.isclose(out.jac[1].toarray()[0, 0], 0.0), "no d/dRs when saturated"
        print("ok test_saturated_property_does_not_depend_on_the_key")

    def test_undersaturated_carries_both_derivatives() -> None:
        from pymrst.utils.adi import init_variables_adi
        tab = _table()
        p, rs = init_variables_adi(np.array([3.0]), np.array([0.5]))
        out = interp_reg_pvt([tab], p, rs, np.array([False]), [ALL])
        assert isinstance(out, ADI)
        assert abs(out.jac[1].toarray()[0, 0]) > 0, "d/dRs must be non-zero"
        print("ok test_undersaturated_carries_both_derivatives")

    def test_regions_pick_their_own_tables() -> None:
        base = _table()
        shifted = preprocess_table_pvt(PackedPVT(
            key=np.array([0.0, 1.0]), pos=np.array([0, 2, 4]),
            data=np.array([[1.0, 101.0], [3.0, 103.0],
                           [2.0, 112.0], [3.0, 113.0]])))
        out = interp_reg_pvt([base, shifted], np.array([1.0, 1.0]),
                             np.array([0.0, 0.0]), np.array([True, True]),
                             [np.array([0]), np.array([1])])
        assert np.allclose(out, [1.0, 101.0]), out
        print("ok test_regions_pick_their_own_tables")

    test_saturated_cells_follow_the_bubble_point_locus()
    test_undersaturated_cells_use_the_two_dimensional_branch()
    test_the_flag_selects_between_the_two_regimes()
    test_mixed_flags_are_handled_cell_by_cell()
    test_saturated_property_does_not_depend_on_the_key()
    test_undersaturated_carries_both_derivatives()
    test_regions_pick_their_own_tables()
    print("\nAll tests passed!")
