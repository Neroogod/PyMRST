"""Gas-water saturation functions from SGWFN (MRST ``props/assignSGWFN.m``).

SGWFN describes a gas-water system with no oil: Sg, krG, krW, Pcwg. One table
supplies both relative permeabilities, so the water curve has to be turned
round - it is tabulated against Sg but asked for at Sw = 1 - Sg, which also
restores an increasing abscissa.

Unlike SWOF and SGOF, the capillary pressure is always attached. There is no
"was one stated" test in MRST here, and adding one would change behaviour for
a deck that states an all-zero column.
"""

from typing import Any, Dict

import numpy as np

from pymrst.modules.ad_props.utils.extend_tab import extend_tab
from pymrst.modules.ad_props.utils.get_reg_map import get_reg_map
from pymrst.modules.ad_props.utils.interp_reg import interp_reg


def assign_sgwfn(f: Dict[str, Any], sgwfn, reg) -> Dict[str, Any]:
    """Store ``krG``, ``krW`` and ``pcWG`` as region-dispatching closures."""
    tables = [np.asarray(t, dtype=float) for t in sgwfn]

    kr_g = extend_tab([t[:, [0, 1]] for t in tables])
    # water is tabulated against Sg; re-index on Sw = 1 - Sg and reverse
    kr_w = extend_tab([np.column_stack([1.0 - t[:, 0], t[:, 2]])[::-1]
                       for t in tables])
    pc_wg = extend_tab([t[:, [0, 3]] for t in tables])

    def _dispatch(table_set):
        def evaluate(s, cell_inx=None, _t=table_set):
            index = get_reg_map(s, reg.SATNUM, reg.SATINX, cell_inx=cell_inx)
            return interp_reg(_t, s, index)
        return evaluate

    f["krG"] = _dispatch(kr_g)
    f["krW"] = _dispatch(kr_w)
    f["pcWG"] = _dispatch(pc_wg)
    return f


assignSGWFN = assign_sgwfn


if __name__ == "__main__":
    from pymrst.modules.ad_props.utils.handle_regions import Regions

    #        Sg    krG   krW   Pcwg
    TABLE = np.array([[0.00, 0.00, 1.00, 0.0],
                      [0.30, 0.10, 0.50, 2.0e3],
                      [0.70, 0.60, 0.10, 8.0e3],
                      [1.00, 1.00, 0.00, 2.0e4]])

    def test_krg_is_read_at_gas_saturation() -> None:
        f = assign_sgwfn({}, [TABLE], Regions())
        assert np.allclose(f["krG"](np.array([0.30])), 0.10)
        print("ok test_krg_is_read_at_gas_saturation")

    def test_krw_is_read_at_water_saturation() -> None:
        """The reversal: krW(Sw), from a table written against Sg."""
        f = assign_sgwfn({}, [TABLE], Regions())
        assert np.allclose(f["krW"](np.array([1.0 - 0.30])), 0.50)
        print("ok test_krw_is_read_at_water_saturation")

    def test_krw_is_not_accidentally_a_function_of_sg() -> None:
        f = assign_sgwfn({}, [TABLE], Regions())
        assert not np.allclose(f["krW"](np.array([0.30])), 0.50)
        print("ok test_krw_is_not_accidentally_a_function_of_sg")

    def test_capillary_pressure_is_read_at_gas_saturation() -> None:
        f = assign_sgwfn({}, [TABLE], Regions())
        assert np.allclose(f["pcWG"](np.array([0.70])), 8.0e3)
        print("ok test_capillary_pressure_is_read_at_gas_saturation")

    def test_capillary_pressure_is_always_attached() -> None:
        """No 'was one stated' test here, unlike SWOF."""
        table = TABLE.copy()
        table[:, 3] = 0.0
        f = assign_sgwfn({}, [table], Regions())
        assert "pcWG" in f and np.allclose(f["pcWG"](np.array([0.30])), 0.0)
        print("ok test_capillary_pressure_is_always_attached")

    def test_curves_are_flat_outside_the_table() -> None:
        f = assign_sgwfn({}, [TABLE], Regions())
        assert np.allclose(f["krG"](np.array([-0.2])), 0.0)
        assert np.allclose(f["krG"](np.array([1.5])), 1.0)
        print("ok test_curves_are_flat_outside_the_table")

    def test_one_call_covers_every_region() -> None:
        second = TABLE.copy()
        second[:, 1] *= 0.5
        reg = Regions(sat=2, SATNUM=np.array([0, 1]),
                      SATINX=[np.array([0]), np.array([1])])
        f = assign_sgwfn({}, [TABLE, second], reg)
        got = f["krG"](np.full(2, 0.30))
        assert np.allclose(got, [0.10, 0.05]), got
        print("ok test_one_call_covers_every_region")

    test_krg_is_read_at_gas_saturation()
    test_krw_is_read_at_water_saturation()
    test_krw_is_not_accidentally_a_function_of_sg()
    test_capillary_pressure_is_read_at_gas_saturation()
    test_capillary_pressure_is_always_attached()
    test_curves_are_flat_outside_the_table()
    test_one_call_covers_every_region()
    print("\nAll tests passed!")
