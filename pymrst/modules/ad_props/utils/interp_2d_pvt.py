"""Undersaturated PVT interpolation (MRST ``ad-props/utils/interp2DPVT.m``).

The undersaturated side of a PVTO/PVTG table: a pressure lookup on each of the
two branches bracketing the key (Rs for live oil, Rv for wet gas), blended
linearly in the key. It differs from `interp_2d_table` in that the branches
need not share an abscissa, so each is interpolated on its own `x`.

Region-aware: one packed table per region, as MRST's `T{k}`.
"""

from typing import List, Sequence

import numpy as np

from pymrst.modules.ad_props.utils.get_reg_map import ALL
from pymrst.modules.ad_props.utils.preprocess_table_pvt import (
    PackedPVT,
    preprocess_table_pvt,
)
from pymrst.utils.adi import ADI, _diag_mul, ad_from_blocks, is_ad, value
from pymrst.utils.interp_table import _interp1


def _as_expanded(table) -> PackedPVT:
    if getattr(table, "expanded", None):
        return table
    return preprocess_table_pvt(table)


def _kernel(table: PackedPVT, xi: np.ndarray, vi: np.ndarray, column: int):
    """Value and both partials for one region's table."""
    key = np.asarray(table.key, dtype=float).ravel()
    # end intervals are open, so a key beyond the table clamps into them
    bin_index = np.clip(np.searchsorted(key, vi, side="right") - 1,
                        0, max(key.size - 2, 0))
    span = key[bin_index + 1] - key[bin_index]
    weight = (vi - key[bin_index]) / span
    dweight = 1.0 / span

    left = np.zeros(xi.shape)
    right = np.zeros(xi.shape)
    dleft = np.zeros(xi.shape)
    dright = np.zeros(xi.shape)
    for branch in range(key.size - 1):
        selected = bin_index == branch
        if not np.any(selected):
            continue
        lo, hi = table.expanded[branch], table.expanded[branch + 1]
        left[selected], dleft[selected] = _interp1(lo.x, lo.F[:, column],
                                                   xi[selected])
        right[selected], dright[selected] = _interp1(hi.x, hi.F[:, column],
                                                     xi[selected])

    values = left * (1.0 - weight) + right * weight
    dx = dleft * (1.0 - weight) + dright * weight
    dv = -(left - right) * dweight
    return values, dx, dv


def interp_2d_pvt(tables: Sequence[PackedPVT], x, v, reg_inx: Sequence,
                  *, column: int = 0):
    """Interpolate the undersaturated branches at ``(x, v)``, per region."""
    xi = np.atleast_1d(value(x)).astype(float)
    vi = np.atleast_1d(value(v)).astype(float)
    values = np.zeros(xi.shape)
    dx = np.zeros(xi.shape)
    dv = np.zeros(xi.shape)

    for region, subs in enumerate(reg_inx):
        table = _as_expanded(tables[region])
        if subs is ALL or isinstance(subs, slice):
            values, dx, dv = _kernel(table, xi, vi, column)
            break
        subs = np.asarray(subs)
        if subs.size == 0:
            continue
        values[subs], dx[subs], dv[subs] = _kernel(table, xi[subs], vi[subs],
                                                   column)

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


interp2DPVT = interp_2d_pvt


if __name__ == "__main__":

    def _table():
        """Two branches whose property is exactly `x + 10*key`."""
        data = np.array([
            [1.0, 1.0], [2.0, 2.0],            # key 0: F = x
            [1.0, 11.0], [2.0, 12.0],          # key 1: F = x + 10
        ])
        return preprocess_table_pvt(
            PackedPVT(key=np.array([0.0, 1.0]), pos=np.array([0, 2, 4]),
                      data=data))

    def test_reproduces_the_branch_values() -> None:
        tab = _table()
        out = interp_2d_pvt([tab], np.array([1.0, 2.0]), np.array([0.0, 0.0]),
                            [ALL])
        assert np.allclose(out, [1.0, 2.0]), out
        print("ok test_reproduces_the_branch_values")

    def test_blends_linearly_between_branches() -> None:
        tab = _table()
        out = interp_2d_pvt([tab], np.array([1.5, 1.5, 1.5]),
                            np.array([0.0, 0.5, 1.0]), [ALL])
        assert np.allclose(out, [1.5, 6.5, 11.5]), out
        print("ok test_blends_linearly_between_branches")

    def test_partials_match_the_analytic_field() -> None:
        tab = _table()
        _, dx, dv = _kernel(tab, np.array([1.5]), np.array([0.5]), 0)
        assert np.isclose(dx[0], 1.0), dx      # d(x + 10k)/dx
        assert np.isclose(dv[0], 10.0), dv     # d(x + 10k)/dk
        print("ok test_partials_match_the_analytic_field")

    def test_carries_derivatives_through_adi() -> None:
        from pymrst.utils.adi import init_variables_adi
        tab = _table()
        p, rs = init_variables_adi(np.array([1.5]), np.array([0.5]))
        out = interp_2d_pvt([tab], p, rs, [ALL])
        assert isinstance(out, ADI)
        assert np.isclose(out.jac[0].toarray()[0, 0], 1.0)
        assert np.isclose(out.jac[1].toarray()[0, 0], 10.0)
        print("ok test_carries_derivatives_through_adi")

    def test_each_region_uses_its_own_table() -> None:
        base = _table()
        shifted = preprocess_table_pvt(PackedPVT(
            key=np.array([0.0, 1.0]), pos=np.array([0, 2, 4]),
            data=np.array([[1.0, 101.0], [2.0, 102.0],
                           [1.0, 111.0], [2.0, 112.0]])))
        out = interp_2d_pvt([base, shifted], np.array([1.0, 1.0]),
                            np.array([0.0, 0.0]),
                            [np.array([0]), np.array([1])])
        assert np.allclose(out, [1.0, 101.0]), out
        print("ok test_each_region_uses_its_own_table")

    test_reproduces_the_branch_values()
    test_blends_linearly_between_branches()
    test_partials_match_the_analytic_field()
    test_carries_derivatives_through_adi()
    test_each_region_uses_its_own_table()
    print("\nAll tests passed!")
