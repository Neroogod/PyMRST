"""Water viscosity against temperature from WATERVIST (MRST ``props/assignWATERVIST.m``).

WATERVIST gives water viscosity against temperature for a thermal run.

It partitions on the **saturation** regions, not the PVT ones, which is
surprising for a fluid property. MRST names the index ``reg.SATKINX``,
a field `handleRegions` never creates - so this line errors in MRST as
written. This port uses ``SATINX``, the field that exists and that the
neighbouring thermal keywords use.
"""

from typing import Any, Dict

import numpy as np

from pymrst.modules.ad_props.utils.extend_tab import extend_tab
from pymrst.modules.ad_props.utils.get_reg_map import get_reg_map
from pymrst.modules.ad_props.utils.interp_reg import interp_reg


def assign_watervist(f: Dict[str, Any], watervist, reg) -> Dict[str, Any]:
    """Store ``muWT`` as a region-dispatching closure."""
    tables = extend_tab([np.asarray(t, dtype=float)[:, [0, 1]] for t in watervist])

    def evaluate(x, cell_inx=None, _t=tables):
        index = get_reg_map(x, reg.SATNUM, reg.SATINX, cell_inx=cell_inx)
        return interp_reg(_t, x, index)

    f["muWT"] = evaluate
    return f


assignWATERVIST = assign_watervist


if __name__ == "__main__":
    from pymrst.modules.ad_props.utils.handle_regions import Regions

    TABLE = np.array([[300.0, 8.0e-4],
                      [350.0, 4.0e-4],
                      [400.0, 2.0e-4]])

    def test_the_value_is_read_from_the_table() -> None:
        f = assign_watervist({}, [TABLE], Regions())
        assert np.allclose(f["muWT"](np.array([350.0])), 4.0e-4)
        print("ok test_the_value_is_read_from_the_table")

    def test_it_is_flat_outside_the_table() -> None:
        f = assign_watervist({}, [TABLE], Regions())
        below = f["muWT"](np.array([TABLE[0, 0] - 5.0]))
        assert np.allclose(below, TABLE[0, 1])
        print("ok test_it_is_flat_outside_the_table")

    def test_it_is_a_closure_not_a_list() -> None:
        f = assign_watervist({}, [TABLE], Regions())
        assert callable(f["muWT"]) and not isinstance(f["muWT"], list)
        print("ok test_it_is_a_closure_not_a_list")

    def test_one_call_covers_every_region() -> None:
        second = TABLE.copy()
        second[:, 1] *= 2.0
        reg = Regions(SATNUM=np.array([0, 1]),
                      SATINX=[np.array([0]), np.array([1])])
        f = assign_watervist({}, [TABLE, second], reg)
        got = f["muWT"](np.full(2, 350.0))
        assert np.allclose(got, [4.0e-4, 4.0e-4 * 2.0]), got
        print("ok test_one_call_covers_every_region")


    test_the_value_is_read_from_the_table()
    test_it_is_flat_outside_the_table()
    test_it_is_a_closure_not_a_list()
    test_one_call_covers_every_region()
    print("\nAll tests passed!")
