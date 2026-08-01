"""Oil viscosity against temperature from OILVIST (MRST ``props/assignOILVIST.m``).

OILVIST is WATERVIST's oil counterpart, and it matters more: heavy oil
viscosity falls by orders of magnitude over a few tens of degrees, which
is the mechanism a steam flood depends on.

It carries the same ``reg.SATKINX`` defect as WATERVIST; see that file.
"""

from typing import Any, Dict

import numpy as np

from pymrst.modules.ad_props.utils.extend_tab import extend_tab
from pymrst.modules.ad_props.utils.get_reg_map import get_reg_map
from pymrst.modules.ad_props.utils.interp_reg import interp_reg


def assign_oilvist(f: Dict[str, Any], oilvist, reg) -> Dict[str, Any]:
    """Store ``muOT`` as a region-dispatching closure."""
    tables = extend_tab([np.asarray(t, dtype=float)[:, [0, 1]] for t in oilvist])

    def evaluate(x, cell_inx=None, _t=tables):
        index = get_reg_map(x, reg.SATNUM, reg.SATINX, cell_inx=cell_inx)
        return interp_reg(_t, x, index)

    f["muOT"] = evaluate
    return f


assignOILVIST = assign_oilvist


if __name__ == "__main__":
    from pymrst.modules.ad_props.utils.handle_regions import Regions

    TABLE = np.array([[300.0, 1.0],
                      [350.0, 0.05],
                      [400.0, 0.005]])

    def test_the_value_is_read_from_the_table() -> None:
        f = assign_oilvist({}, [TABLE], Regions())
        assert np.allclose(f["muOT"](np.array([350.0])), 0.05)
        print("ok test_the_value_is_read_from_the_table")

    def test_it_is_flat_outside_the_table() -> None:
        f = assign_oilvist({}, [TABLE], Regions())
        below = f["muOT"](np.array([TABLE[0, 0] - 5.0]))
        assert np.allclose(below, TABLE[0, 1])
        print("ok test_it_is_flat_outside_the_table")

    def test_it_is_a_closure_not_a_list() -> None:
        f = assign_oilvist({}, [TABLE], Regions())
        assert callable(f["muOT"]) and not isinstance(f["muOT"], list)
        print("ok test_it_is_a_closure_not_a_list")

    def test_one_call_covers_every_region() -> None:
        second = TABLE.copy()
        second[:, 1] *= 2.0
        reg = Regions(SATNUM=np.array([0, 1]),
                      SATINX=[np.array([0]), np.array([1])])
        f = assign_oilvist({}, [TABLE, second], reg)
        got = f["muOT"](np.full(2, 350.0))
        assert np.allclose(got, [0.05, 0.05 * 2.0]), got
        print("ok test_one_call_covers_every_region")


    test_the_value_is_read_from_the_table()
    test_it_is_flat_outside_the_table()
    test_it_is_a_closure_not_a_list()
    test_one_call_covers_every_region()
    print("\nAll tests passed!")
