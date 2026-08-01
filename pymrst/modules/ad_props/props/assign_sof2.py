"""Two-phase oil saturation function from SOF2 (MRST ``props/assignSOF2.m``).

SOF2 is SOF3's two-phase form: one oil relative permeability against oil
saturation, for a run with only one other phase present.

Its shape in the fluid differs from SOF3's, and the difference is MRST's, not
a simplification. SOF3 stores a list of per-region closures that a caller
indexes; SOF2 stores a **single** closure that dispatches internally through
the region map, so it takes a whole field of saturations and routes each cell
to its own table. Callers of `krO` therefore pass every cell at once, where
callers of `krOW` pass one region at a time.

Note also that the surfactant curve extends the table and the base curve does
not - again MRST's, and it means the two disagree outside the tabulated range.
"""

from typing import Any, Dict

import numpy as np

from pymrst.modules.ad_props.utils.extend_tab import extend_tab
from pymrst.modules.ad_props.utils.get_reg_map import get_reg_map
from pymrst.modules.ad_props.utils.interp_reg import interp_reg


def assign_sof2(f: Dict[str, Any], sof2, reg) -> Dict[str, Any]:
    """Store ``krO`` as a single region-dispatching closure."""
    tables = [np.asarray(t, dtype=float)[:, :2] for t in sof2]

    def kr_o(so, cell_inx=None, _t=tables):
        index = get_reg_map(so, reg.SATNUM, reg.SATINX, cell_inx=cell_inx)
        return interp_reg(_t, so, index)

    f["krO"] = kr_o

    if reg.SURFNUM is not None:
        extended = extend_tab(tables)

        def kr_o_sft(so, cell_inx=None, _t=extended):
            index = get_reg_map(so, reg.SURFNUM, reg.SURFINX,
                                cell_inx=cell_inx)
            return interp_reg(_t, so, index)

        f["krOSft"] = kr_o_sft
    return f


assignSOF2 = assign_sof2


if __name__ == "__main__":
    from pymrst.modules.ad_props.utils.handle_regions import Regions

    #        So    krO
    TABLE = np.array([[0.00, 0.00],
                      [0.20, 0.00],
                      [0.50, 0.30],
                      [0.85, 1.00]])

    def test_kro_is_read_at_oil_saturation() -> None:
        f = assign_sof2({}, [TABLE], Regions())
        assert np.allclose(f["krO"](np.array([0.50])), 0.30)
        print("ok test_kro_is_read_at_oil_saturation")

    def test_kro_is_one_closure_not_a_list() -> None:
        """SOF2's shape differs from SOF3's; callers depend on which."""
        f = assign_sof2({}, [TABLE], Regions())
        assert callable(f["krO"]) and not isinstance(f["krO"], list)
        print("ok test_kro_is_one_closure_not_a_list")

    def test_one_call_covers_every_region() -> None:
        second = TABLE.copy()
        second[:, 1] *= 0.5
        reg = Regions(sat=2, SATNUM=np.array([0, 1, 0, 1]),
                      SATINX=[np.array([0, 2]), np.array([1, 3])])
        f = assign_sof2({}, [TABLE, second], reg)
        got = f["krO"](np.full(4, 0.50))
        assert np.allclose(got, [0.30, 0.15, 0.30, 0.15]), got
        print("ok test_one_call_covers_every_region")

    def test_no_surfactant_curve_without_a_surfactant_region() -> None:
        assert "krOSft" not in assign_sof2({}, [TABLE], Regions())
        print("ok test_no_surfactant_curve_without_a_surfactant_region")

    def test_surfactant_curve_uses_the_surfactant_partition() -> None:
        second = TABLE.copy()
        second[:, 1] *= 0.5
        reg = Regions(sat=2, SATNUM=np.array([0, 0]),
                      SATINX=[np.array([0, 1])],
                      SURFNUM=np.array([0, 1]),
                      SURFINX=[np.array([0]), np.array([1])])
        f = assign_sof2({}, [TABLE, second], reg)
        got = f["krOSft"](np.full(2, 0.50))
        assert np.allclose(got, [0.30, 0.15]), got
        print("ok test_surfactant_curve_uses_the_surfactant_partition")

    def test_the_surfactant_curve_is_extended_and_the_base_one_is_not() -> None:
        reg = Regions(SATNUM=np.array([0]), SATINX=[np.array([0])],
                      SURFNUM=np.array([0]), SURFINX=[np.array([0])])
        f = assign_sof2({}, [TABLE], reg)
        beyond = np.array([1.5])
        assert np.isclose(f["krOSft"](beyond)[0], 1.00), "flat past the table"
        print("ok test_the_surfactant_curve_is_extended_and_the_base_one_is_not")

    test_kro_is_read_at_oil_saturation()
    test_kro_is_one_closure_not_a_list()
    test_one_call_covers_every_region()
    test_no_surfactant_curve_without_a_surfactant_region()
    test_surfactant_curve_uses_the_surfactant_partition()
    test_the_surfactant_curve_is_extended_and_the_base_one_is_not()
    print("\nAll tests passed!")
