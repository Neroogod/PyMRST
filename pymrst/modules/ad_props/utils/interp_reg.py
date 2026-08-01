"""Region-wise table interpolation (MRST ``interpReg.m`` and ``dinterpReg.m``).

One table per region, one lookup per region, results scattered back into the
caller's ordering. This is the shape every regional property function takes:
`assignSWOF` and friends close over a table list and call in here.

The single-region case is kept separate rather than folded into the loop,
because it is overwhelmingly the common one and MRST separates it too.
"""

from typing import List, Optional, Sequence, Tuple

import numpy as np

from pymrst.modules.ad_props.utils.get_reg_map import ALL
from pymrst.utils.interp_table import _interp1


def _table_columns(table) -> Tuple[np.ndarray, np.ndarray]:
    table = np.asarray(table, dtype=float)
    return table[:, 0], table[:, 1]


def interp_reg(
    tables: Sequence[np.ndarray],
    xi: np.ndarray,
    reg_inx: Sequence,
    *,
    derivative: bool = False,
):
    """Interpolate ``xi`` through one table per region.

    ``tables[k]`` is the two-column table for region ``k`` and ``reg_inx[k]``
    the indices of ``xi`` belonging to it. With ``derivative=True`` the slope
    is returned alongside the value.
    """
    xi = np.asarray(xi, dtype=float)
    if xi.size == 0:
        empty = np.zeros(0)
        return (empty, empty) if derivative else empty

    values = np.zeros(xi.shape)
    slopes = np.zeros(xi.shape)

    if len(reg_inx) == 1 and (reg_inx[0] is ALL or isinstance(reg_inx[0], slice)):
        x_tab, f_tab = _table_columns(tables[0])
        values, slopes = _interp1(x_tab, f_tab, xi)
        return (values, slopes) if derivative else values

    for region, subs in enumerate(reg_inx):
        subs = np.asarray(subs)
        if subs.size == 0:
            continue
        x_tab, f_tab = _table_columns(tables[region])
        values[subs], slopes[subs] = _interp1(x_tab, f_tab, xi[subs])
    return (values, slopes) if derivative else values


def dinterp_reg(tables: Sequence[np.ndarray], xi: np.ndarray,
                reg_inx: Sequence) -> np.ndarray:
    """The derivative of `interp_reg`, MRST's ``dinterpReg``."""
    return interp_reg(tables, xi, reg_inx, derivative=True)[1]


interpReg = interp_reg
dinterpReg = dinterp_reg


if __name__ == "__main__":

    LINEAR = np.array([[0.0, 0.0], [1.0, 10.0], [2.0, 30.0]])
    STEEP = np.array([[0.0, 0.0], [1.0, 100.0], [2.0, 300.0]])

    def test_single_region_uses_the_only_table() -> None:
        out = interp_reg([LINEAR], np.array([0.5, 1.5]), [ALL])
        assert np.allclose(out, [5.0, 20.0]), out
        print("ok test_single_region_uses_the_only_table")

    def test_each_region_uses_its_own_table() -> None:
        xi = np.array([0.5, 0.5, 0.5, 0.5])
        reg_inx = [np.array([0, 1]), np.array([2, 3])]
        out = interp_reg([LINEAR, STEEP], xi, reg_inx)
        assert np.allclose(out[:2], 5.0), out
        assert np.allclose(out[2:], 50.0), out
        print("ok test_each_region_uses_its_own_table")

    def test_results_land_in_the_callers_ordering() -> None:
        """Region indices need not be contiguous; the scatter must respect that."""
        xi = np.array([0.5, 0.5, 0.5, 0.5])
        reg_inx = [np.array([0, 3]), np.array([1, 2])]
        out = interp_reg([LINEAR, STEEP], xi, reg_inx)
        assert np.allclose(out, [5.0, 50.0, 50.0, 5.0]), out
        print("ok test_results_land_in_the_callers_ordering")

    def test_derivative_matches_the_table_slope() -> None:
        xi = np.array([0.5, 1.5])
        value, slope = interp_reg([LINEAR], xi, [ALL], derivative=True)
        assert np.allclose(slope, [10.0, 20.0]), slope
        assert np.allclose(dinterp_reg([LINEAR], xi, [ALL]), slope)
        print("ok test_derivative_matches_the_table_slope")

    def test_derivative_is_per_region_too() -> None:
        xi = np.array([0.5, 0.5])
        reg_inx = [np.array([0]), np.array([1])]
        slope = dinterp_reg([LINEAR, STEEP], xi, reg_inx)
        assert np.allclose(slope, [10.0, 100.0]), slope
        print("ok test_derivative_is_per_region_too")

    def test_empty_input_returns_empty() -> None:
        out = interp_reg([LINEAR], np.zeros(0), [ALL])
        assert out.size == 0
        print("ok test_empty_input_returns_empty")

    def test_an_empty_region_is_skipped() -> None:
        xi = np.array([0.5, 0.5])
        out = interp_reg([LINEAR, STEEP], xi,
                         [np.array([0, 1]), np.zeros(0, dtype=int)])
        assert np.allclose(out, 5.0), out
        print("ok test_an_empty_region_is_skipped")

    test_single_region_uses_the_only_table()
    test_each_region_uses_its_own_table()
    test_results_land_in_the_callers_ordering()
    test_derivative_matches_the_table_slope()
    test_derivative_is_per_region_too()
    test_empty_input_returns_empty()
    test_an_empty_region_is_skipped()
    print("\nAll tests passed!")
