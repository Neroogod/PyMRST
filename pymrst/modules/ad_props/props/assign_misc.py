"""Solvent miscibility from MISC (MRST ``props/assignMISC.m``).

MISC gives the miscibility of a solvent against the solvent fraction of
the gas phase: 1 means fully miscible, so solvent and oil mix without an
interface.

The field is ``Ms``, and it multiplies with PMISC's ``Mp`` rather than
replacing it - a deck may state both, and each answers a different
question: MISC how much solvent is present, PMISC whether the pressure
is high enough for it to matter.
"""

from typing import Any, Dict

import numpy as np

from pymrst.modules.ad_props.utils.extend_tab import extend_tab
from pymrst.modules.ad_props.utils.get_reg_map import get_reg_map
from pymrst.modules.ad_props.utils.interp_reg import interp_reg


def assign_misc(f: Dict[str, Any], misc, reg) -> Dict[str, Any]:
    """Store ``Ms`` as a region-dispatching closure."""
    tables = extend_tab([np.asarray(t, dtype=float)[:, [0, 1]] for t in misc])

    def evaluate(x, cell_inx=None, _t=tables):
        index = get_reg_map(x, reg.SATNUM, reg.SATINX, cell_inx=cell_inx)
        return interp_reg(_t, x, index)

    f["Ms"] = evaluate
    return f


assignMISC = assign_misc


if __name__ == "__main__":
    from pymrst.modules.ad_props.utils.handle_regions import Regions

    TABLE = np.array([[0.0, 0.0],
                      [0.5, 0.6],
                      [1.0, 1.0]])

    def test_the_value_is_read_from_the_table() -> None:
        f = assign_misc({}, [TABLE], Regions())
        assert np.allclose(f["Ms"](np.array([0.5])), 0.6)
        print("ok test_the_value_is_read_from_the_table")

    def test_it_is_flat_outside_the_table() -> None:
        f = assign_misc({}, [TABLE], Regions())
        below = f["Ms"](np.array([TABLE[0, 0] - 5.0]))
        assert np.allclose(below, TABLE[0, 1])
        print("ok test_it_is_flat_outside_the_table")

    def test_it_is_a_closure_not_a_list() -> None:
        f = assign_misc({}, [TABLE], Regions())
        assert callable(f["Ms"]) and not isinstance(f["Ms"], list)
        print("ok test_it_is_a_closure_not_a_list")

    def test_one_call_covers_every_region() -> None:
        second = TABLE.copy()
        second[:, 1] *= 2.0
        reg = Regions(SATNUM=np.array([0, 1]),
                      SATINX=[np.array([0]), np.array([1])])
        f = assign_misc({}, [TABLE, second], reg)
        got = f["Ms"](np.full(2, 0.5))
        assert np.allclose(got, [0.6, 0.6 * 2.0]), got
        print("ok test_one_call_covers_every_region")


    test_the_value_is_read_from_the_table()
    test_it_is_flat_outside_the_table()
    test_it_is_a_closure_not_a_list()
    test_one_call_covers_every_region()
    print("\nAll tests passed!")
