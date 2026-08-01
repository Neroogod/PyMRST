"""Solvent saturation functions from SSFN (MRST ``props/assignSSFN.m``).

SSFN gives the fractional-flow curves for a solvent flood: gas and
solvent relative permeability against the solvent-plus-gas saturation.
Both are read at the same abscissa, so no re-indexing is needed.

The curves are stored as single closures that dispatch through the saturation
region map, so one call covers the whole field. That differs from SWOF's
per-region lists; the shape is MRST's and callers depend on which they get.
"""

from typing import Any, Dict

import numpy as np

from pymrst.modules.ad_props.utils.extend_tab import extend_tab
from pymrst.modules.ad_props.utils.get_reg_map import get_reg_map
from pymrst.modules.ad_props.utils.interp_reg import interp_reg


def assign_ssfn(f: Dict[str, Any], ssfn, reg) -> Dict[str, Any]:
    """Store ``krFG`` and ``krFS``."""
    tables = [np.asarray(t, dtype=float) for t in ssfn]

    def _dispatch(column):
        curves = extend_tab([t[:, [0, column]] for t in tables])

        def evaluate(s, cell_inx=None, _t=curves):
            index = get_reg_map(s, reg.SATNUM, reg.SATINX, cell_inx=cell_inx)
            return interp_reg(_t, s, index)
        return evaluate

    f["krFG"] = _dispatch(1)
    f["krFS"] = _dispatch(2)
    return f


assignSSFN = assign_ssfn


if __name__ == "__main__":
    from pymrst.modules.ad_props.utils.handle_regions import Regions

    TABLE = np.array([[0.00, 0.00, 1.00],
                      [0.40, 0.35, 0.60],
                      [1.00, 1.00, 0.00]])

    def test_first_curve_is_read_from_column_one() -> None:
        f = assign_ssfn({}, [TABLE], Regions())
        assert np.allclose(f["krFG"](np.array([0.40])), 0.35)
        print("ok test_first_curve_is_read_from_column_one")

    def test_second_curve_is_read_from_column_two() -> None:
        f = assign_ssfn({}, [TABLE], Regions())
        assert np.allclose(f["krFS"](np.array([0.40])), 0.60)
        print("ok test_second_curve_is_read_from_column_two")

    def test_the_two_curves_are_distinct() -> None:
        f = assign_ssfn({}, [TABLE], Regions())
        probe = np.array([0.40])
        assert not np.allclose(f["krFG"](probe), f["krFS"](probe))
        print("ok test_the_two_curves_are_distinct")

    def test_curves_are_flat_outside_the_table() -> None:
        f = assign_ssfn({}, [TABLE], Regions())
        assert np.allclose(f["krFG"](np.array([-0.5])), TABLE[0, 1])
        assert np.allclose(f["krFG"](np.array([2.0])), TABLE[-1, 1])
        print("ok test_curves_are_flat_outside_the_table")

    def test_one_call_covers_every_region() -> None:
        second = TABLE.copy()
        second[:, 1] *= 0.5
        reg = Regions(sat=2, SATNUM=np.array([0, 1]),
                      SATINX=[np.array([0]), np.array([1])])
        f = assign_ssfn({}, [TABLE, second], reg)
        got = f["krFG"](np.full(2, 0.40))
        assert np.allclose(got, [0.35, 0.35 * 0.5]), got
        print("ok test_one_call_covers_every_region")

    def test_curves_are_closures_not_lists() -> None:
        f = assign_ssfn({}, [TABLE], Regions())
        assert callable(f["krFG"]) and not isinstance(f["krFG"], list)
        print("ok test_curves_are_closures_not_lists")

    test_first_curve_is_read_from_column_one()
    test_second_curve_is_read_from_column_two()
    test_the_two_curves_are_distinct()
    test_curves_are_flat_outside_the_table()
    test_one_call_covers_every_region()
    test_curves_are_closures_not_lists()
    print("\nAll tests passed!")
