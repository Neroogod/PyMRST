"""Solvent PVT from PVDS (MRST ``props/assignPVDS.m``).

PVDS is PVDG's solvent counterpart: pressure, formation volume factor and
viscosity for the injected solvent, which is treated as a fourth phase with no
dissolution.

Two differences from `assign_pvdg` are deliberate and both are MRST's:

- the table is **not** extended down to zero pressure, so a solvent property
  evaluated below the first tabulated pressure follows whatever the
  interpolator does at the boundary rather than a flat guard;
- the properties are single dispatching closures partitioned on PVTNUM,
  where PVDG produces one closure per region.
"""

from typing import Any, Dict

import numpy as np

from pymrst.modules.ad_props.utils.get_reg_map import get_reg_map
from pymrst.modules.ad_props.utils.interp_reg import interp_reg


def assign_pvds(f: Dict[str, Any], pvds, reg) -> Dict[str, Any]:
    """Store ``bS`` and ``muS`` as region-dispatching closures."""
    tables = [np.asarray(t, dtype=float) for t in pvds]
    b_tables = [np.column_stack([t[:, 0], 1.0 / t[:, 1]]) for t in tables]
    mu_tables = [t[:, [0, 2]] for t in tables]

    def _dispatch(table_set):
        def evaluate(p, cell_inx=None, _t=table_set):
            index = get_reg_map(p, reg.PVTNUM, reg.PVTINX, cell_inx=cell_inx)
            return interp_reg(_t, p, index)
        return evaluate

    f["bS"] = _dispatch(b_tables)
    f["muS"] = _dispatch(mu_tables)
    return f


assignPVDS = assign_pvds


if __name__ == "__main__":
    from pymrst.modules.ad_props.utils.handle_regions import Regions

    TABLE = np.array([[1.0e7, 5.0e-3, 1.4e-5],
                      [2.0e7, 3.0e-3, 1.7e-5],
                      [3.0e7, 2.5e-3, 2.0e-5]])

    def test_b_is_the_reciprocal_of_the_tabulated_fvf() -> None:
        f = assign_pvds({}, [TABLE], Regions())
        assert np.allclose(f["bS"](np.array([2.0e7])), 1.0 / 3.0e-3)
        print("ok test_b_is_the_reciprocal_of_the_tabulated_fvf")

    def test_viscosity_is_read_straight_from_the_table() -> None:
        f = assign_pvds({}, [TABLE], Regions())
        assert np.allclose(f["muS"](np.array([3.0e7])), 2.0e-5)
        print("ok test_viscosity_is_read_straight_from_the_table")

    def test_b_interpolates_the_reciprocal_not_the_fvf() -> None:
        f = assign_pvds({}, [TABLE], Regions())
        got = f["bS"](np.array([1.5e7]))
        reciprocal = np.interp(1.5e7, TABLE[:, 0], 1.0 / TABLE[:, 1])
        assert np.allclose(got, reciprocal), got
        print("ok test_b_interpolates_the_reciprocal_not_the_fvf")

    def test_the_table_is_not_extended_to_zero_pressure() -> None:
        """PVDG adds a zero-pressure row; PVDS does not, and that is MRST's."""
        from pymrst.modules.ad_props.props.assign_pvdg import assign_pvdg
        g = assign_pvdg({}, [TABLE], Regions())
        assert np.isclose(g["bG"][0](0.0), 1.0 / TABLE[0, 1]), "PVDG is guarded"
        f = assign_pvds({}, [TABLE], Regions())
        assert f["bS"](np.array([0.0])).size == 1, "PVDS still evaluates"
        print("ok test_the_table_is_not_extended_to_zero_pressure")

    def test_properties_are_closures_not_lists() -> None:
        f = assign_pvds({}, [TABLE], Regions())
        assert callable(f["bS"]) and not isinstance(f["bS"], list)
        print("ok test_properties_are_closures_not_lists")

    def test_one_call_covers_every_pvt_region() -> None:
        second = TABLE.copy()
        second[:, 1] *= 2.0
        reg = Regions(pvt=2, PVTNUM=np.array([0, 1]),
                      PVTINX=[np.array([0]), np.array([1])])
        f = assign_pvds({}, [TABLE, second], reg)
        got = f["bS"](np.full(2, 2.0e7))
        assert np.allclose(got, [1 / 3.0e-3, 1 / 6.0e-3]), got
        print("ok test_one_call_covers_every_pvt_region")

    test_b_is_the_reciprocal_of_the_tabulated_fvf()
    test_viscosity_is_read_straight_from_the_table()
    test_b_interpolates_the_reciprocal_not_the_fvf()
    test_the_table_is_not_extended_to_zero_pressure()
    test_properties_are_closures_not_lists()
    test_one_call_covers_every_pvt_region()
    print("\nAll tests passed!")
