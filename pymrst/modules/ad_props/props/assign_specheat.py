"""Fluid internal energy from SPECHEAT (MRST ``props/assignSPECHEAT.m``).

SPECHEAT tabulates each phase's specific heat capacity against temperature -
four columns: temperature, then gas, water and oil.

Two things about the result are easy to get wrong.

**Three fields, not one.** The keyword produces ``uG``, ``uW`` and ``uO``, one
per phase, from columns 2, 3 and 4 in that order. Note the ordering is gas
first, unlike DENSITY's oil-water-gas.

**They are energies, not capacities.** Each looked-up capacity is multiplied by
the temperature. MRST notes in a comment that the energy is properly the
integral of the heat capacity over temperature and that this neglects the
integral; the two agree only when the capacity is constant. It is reproduced
here because changing it would silently move every thermal result away from
MRST's.
"""

from typing import Any, Dict

import numpy as np

from pymrst.modules.ad_props.utils.extend_tab import extend_tab
from pymrst.modules.ad_props.utils.get_reg_map import get_reg_map
from pymrst.modules.ad_props.utils.interp_reg import interp_reg

# Column of SPECHEAT holding each phase's capacity, and the field it becomes.
PHASE_COLUMNS = (("uG", 1), ("uW", 2), ("uO", 3))


def assign_specheat(f: Dict[str, Any], specheat, reg) -> Dict[str, Any]:
    """Store ``uG``, ``uW`` and ``uO`` as region-dispatching closures."""
    tables = [np.asarray(t, dtype=float) for t in specheat]

    for field, column in PHASE_COLUMNS:
        curves = extend_tab([t[:, [0, column]] for t in tables])

        def evaluate(temperature, cell_inx=None, _t=curves):
            index = get_reg_map(temperature, reg.SATNUM, reg.SATINX,
                                cell_inx=cell_inx)
            # capacity * T: MRST's approximation to the energy integral
            return interp_reg(_t, temperature, index) * temperature

        f[field] = evaluate
    return f


assignSPECHEAT = assign_specheat


if __name__ == "__main__":
    from pymrst.modules.ad_props.utils.handle_regions import Regions

    #          T      cG      cW      cO
    TABLE = np.array([[300.0, 1000.0, 4180.0, 1900.0],
                      [350.0, 1100.0, 4200.0, 2000.0],
                      [400.0, 1200.0, 4260.0, 2100.0]])

    def test_all_three_phase_fields_are_created() -> None:
        f = assign_specheat({}, [TABLE], Regions())
        assert set(f) == {"uG", "uW", "uO"}, set(f)
        print("ok test_all_three_phase_fields_are_created")

    def test_the_columns_map_gas_water_oil_in_that_order() -> None:
        """Gas first, unlike DENSITY's oil-water-gas."""
        f = assign_specheat({}, [TABLE], Regions())
        t = np.array([350.0])
        assert np.allclose(f["uG"](t), 1100.0 * 350.0)
        assert np.allclose(f["uW"](t), 4200.0 * 350.0)
        assert np.allclose(f["uO"](t), 2000.0 * 350.0)
        print("ok test_the_columns_map_gas_water_oil_in_that_order")

    def test_the_value_is_an_energy_not_a_capacity() -> None:
        f = assign_specheat({}, [TABLE], Regions())
        got = f["uW"](np.array([350.0]))
        assert np.allclose(got, 4200.0 * 350.0), got
        assert not np.allclose(got, 4200.0), "not the bare capacity"
        print("ok test_the_value_is_an_energy_not_a_capacity")

    def test_energy_rises_with_temperature() -> None:
        f = assign_specheat({}, [TABLE], Regions())
        temperatures = np.array([300.0, 350.0, 400.0])
        assert np.all(np.diff(f["uO"](temperatures)) > 0)
        print("ok test_energy_rises_with_temperature")

    def test_the_capacity_table_is_flat_outside_its_range() -> None:
        """The capacity clamps; the energy still scales with T."""
        f = assign_specheat({}, [TABLE], Regions())
        assert np.allclose(f["uW"](np.array([250.0])), 4180.0 * 250.0)
        print("ok test_the_capacity_table_is_flat_outside_its_range")

    def test_the_fields_are_closures_not_lists() -> None:
        f = assign_specheat({}, [TABLE], Regions())
        assert callable(f["uW"]) and not isinstance(f["uW"], list)
        print("ok test_the_fields_are_closures_not_lists")

    def test_one_call_covers_every_region() -> None:
        second = TABLE.copy()
        second[:, 2] *= 2.0
        reg = Regions(SATNUM=np.array([0, 1]),
                      SATINX=[np.array([0]), np.array([1])])
        f = assign_specheat({}, [TABLE, second], reg)
        got = f["uW"](np.full(2, 350.0))
        assert np.allclose(got, [4200.0 * 350.0, 8400.0 * 350.0]), got
        print("ok test_one_call_covers_every_region")

    def test_each_phase_keeps_its_own_region_tables() -> None:
        second = TABLE.copy()
        second[:, 1] *= 3.0            # gas only
        reg = Regions(SATNUM=np.array([0, 1]),
                      SATINX=[np.array([0]), np.array([1])])
        f = assign_specheat({}, [TABLE, second], reg)
        water = f["uW"](np.full(2, 350.0))
        assert np.allclose(water, 4200.0 * 350.0), "water must be unaffected"
        gas = f["uG"](np.full(2, 350.0))
        assert np.allclose(gas, [1100.0 * 350.0, 3300.0 * 350.0]), gas
        print("ok test_each_phase_keeps_its_own_region_tables")

    test_all_three_phase_fields_are_created()
    test_the_columns_map_gas_water_oil_in_that_order()
    test_the_value_is_an_energy_not_a_capacity()
    test_energy_rises_with_temperature()
    test_the_capacity_table_is_flat_outside_its_range()
    test_the_fields_are_closures_not_lists()
    test_one_call_covers_every_region()
    test_each_phase_keeps_its_own_region_tables()
    print("\nAll tests passed!")
