"""Rock energy from SPECROCK (MRST ``props/assignSPECROCK.m``).

SPECROCK tabulates the rock's specific heat capacity against
temperature, but the field stored is ``eR``, an **energy**: the looked-up
capacity is multiplied by the temperature.

That product is an approximation and MRST says so in a comment - the
energy is properly the integral of the heat capacity over temperature,
and this neglects the integral. The two agree only when the capacity is
constant. It is reproduced here because changing it would silently alter
every thermal result relative to MRST.

It partitions on ROCKNUM, as rock compaction does.
"""

from typing import Any, Dict

import numpy as np

from pymrst.modules.ad_props.utils.extend_tab import extend_tab
from pymrst.modules.ad_props.utils.get_reg_map import get_reg_map
from pymrst.modules.ad_props.utils.interp_reg import interp_reg


def assign_specrock(f: Dict[str, Any], specrock, reg) -> Dict[str, Any]:
    """Store ``eR`` as a region-dispatching closure."""
    tables = extend_tab([np.asarray(t, dtype=float)[:, [0, 1]] for t in specrock])

    def evaluate(x, cell_inx=None, _t=tables):
        index = get_reg_map(x, reg.ROCKNUM, reg.ROCKINX, cell_inx=cell_inx)
        return interp_reg(_t, x, index) * x

    f["eR"] = evaluate
    return f


assignSPECROCK = assign_specrock


if __name__ == "__main__":
    from pymrst.modules.ad_props.utils.handle_regions import Regions

    TABLE = np.array([[300.0, 2.0e6],
                      [350.0, 2.1e6],
                      [400.0, 2.2e6]])

    def test_the_value_is_read_from_the_table() -> None:
        f = assign_specrock({}, [TABLE], Regions())
        assert np.allclose(f["eR"](np.array([350.0])), 2.1e6 * 350.0)
        print("ok test_the_value_is_read_from_the_table")

    def test_it_is_flat_outside_the_table() -> None:
        f = assign_specrock({}, [TABLE], Regions())
        below = f["eR"](np.array([TABLE[0, 0] - 5.0]))
        assert np.allclose(below, TABLE[0, 1] * (TABLE[0, 0] - 5.0))
        print("ok test_it_is_flat_outside_the_table")

    def test_it_is_a_closure_not_a_list() -> None:
        f = assign_specrock({}, [TABLE], Regions())
        assert callable(f["eR"]) and not isinstance(f["eR"], list)
        print("ok test_it_is_a_closure_not_a_list")

    def test_one_call_covers_every_region() -> None:
        second = TABLE.copy()
        second[:, 1] *= 2.0
        reg = Regions(ROCKNUM=np.array([0, 1]),
                      ROCKINX=[np.array([0]), np.array([1])])
        f = assign_specrock({}, [TABLE, second], reg)
        got = f["eR"](np.full(2, 350.0))
        assert np.allclose(got, [2.1e6 * 350.0, 2.1e6 * 350.0 * 2.0]), got
        print("ok test_one_call_covers_every_region")

    def test_the_stored_value_is_an_energy_not_a_capacity() -> None:
        """eR = c(T) * T, MRST's approximation to the integral."""
        f = assign_specrock({}, [TABLE], Regions())
        got = f["eR"](np.array([350.0]))
        assert np.allclose(got, 2.1e6 * 350.0), got
        assert not np.allclose(got, 2.1e6), "not the bare capacity"
        print("ok test_the_stored_value_is_an_energy_not_a_capacity")


    test_the_value_is_read_from_the_table()
    test_it_is_flat_outside_the_table()
    test_it_is_a_closure_not_a_list()
    test_one_call_covers_every_region()
    test_the_stored_value_is_an_energy_not_a_capacity()
    print("\nAll tests passed!")
