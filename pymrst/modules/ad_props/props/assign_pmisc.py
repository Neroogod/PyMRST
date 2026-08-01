"""Pressure-dependent miscibility from PMISC (MRST ``props/assignPMISC.m``).

PMISC gives miscibility against pressure. A solvent flood only becomes
miscible above a minimum miscibility pressure, and this table is where
that threshold lives.

The field is ``Mp``. See `assign_misc` for how the two combine.
"""

from typing import Any, Dict

import numpy as np

from pymrst.modules.ad_props.utils.extend_tab import extend_tab
from pymrst.modules.ad_props.utils.get_reg_map import get_reg_map
from pymrst.modules.ad_props.utils.interp_reg import interp_reg


def assign_pmisc(f: Dict[str, Any], pmisc, reg) -> Dict[str, Any]:
    """Store ``Mp`` as a region-dispatching closure."""
    tables = extend_tab([np.asarray(t, dtype=float)[:, [0, 1]] for t in pmisc])

    def evaluate(x, cell_inx=None, _t=tables):
        index = get_reg_map(x, reg.SATNUM, reg.SATINX, cell_inx=cell_inx)
        return interp_reg(_t, x, index)

    f["Mp"] = evaluate
    return f


assignPMISC = assign_pmisc


if __name__ == "__main__":
    from pymrst.modules.ad_props.utils.handle_regions import Regions

    TABLE = np.array([[1.0e7, 0.0],
                      [2.0e7, 0.5],
                      [3.0e7, 1.0]])

    def test_the_value_is_read_from_the_table() -> None:
        f = assign_pmisc({}, [TABLE], Regions())
        assert np.allclose(f["Mp"](np.array([2.0e7])), 0.5)
        print("ok test_the_value_is_read_from_the_table")

    def test_it_is_flat_outside_the_table() -> None:
        f = assign_pmisc({}, [TABLE], Regions())
        below = f["Mp"](np.array([TABLE[0, 0] - 5.0]))
        assert np.allclose(below, TABLE[0, 1])
        print("ok test_it_is_flat_outside_the_table")

    def test_it_is_a_closure_not_a_list() -> None:
        f = assign_pmisc({}, [TABLE], Regions())
        assert callable(f["Mp"]) and not isinstance(f["Mp"], list)
        print("ok test_it_is_a_closure_not_a_list")

    def test_one_call_covers_every_region() -> None:
        second = TABLE.copy()
        second[:, 1] *= 2.0
        reg = Regions(SATNUM=np.array([0, 1]),
                      SATINX=[np.array([0]), np.array([1])])
        f = assign_pmisc({}, [TABLE, second], reg)
        got = f["Mp"](np.full(2, 2.0e7))
        assert np.allclose(got, [0.5, 0.5 * 2.0]), got
        print("ok test_one_call_covers_every_region")


    test_the_value_is_read_from_the_table()
    test_it_is_flat_outside_the_table()
    test_it_is_a_closure_not_a_list()
    test_one_call_covers_every_region()
    print("\nAll tests passed!")
