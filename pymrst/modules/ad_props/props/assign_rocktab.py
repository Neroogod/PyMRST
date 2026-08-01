"""Tabulated rock compaction from ROCKTAB (MRST ``props/assignROCKTAB.m``).

ROCKTAB is ROCK's table form: pressure against a pore-volume multiplier and a
transmissibility multiplier. It replaces ROCK's closed-form expansion when a
deck states measured compaction behaviour.

Two differences from `assign_rock` are worth being explicit about.

- It partitions on **ROCKNUM**, not PVTNUM. Rock compaction regions follow the
  rock, and a deck can give them a different partition from the fluid's.
- It also supplies ``tranMultR``, which ROCK does not. Compaction closes
  fractures as well as pores, and the two multipliers are independent columns.

`pvMultR` is a single dispatching closure here where `assign_rock` leaves a
bare function or a list. Callers that accept either must handle all three
shapes; that is MRST's arrangement, not something this port introduces.
"""

from typing import Any, Dict

import numpy as np

from pymrst.modules.ad_props.utils.get_reg_map import get_reg_map
from pymrst.modules.ad_props.utils.interp_reg import interp_reg


def assign_rocktab(f: Dict[str, Any], rocktab, reg) -> Dict[str, Any]:
    """Store ``pvMultR`` and ``tranMultR``, partitioned on ROCKNUM."""
    if not isinstance(rocktab, (list, tuple)):
        rocktab = [rocktab]
    tables = [np.asarray(t, dtype=float) for t in rocktab]

    def _dispatch(column):
        curves = [t[:, [0, column]] for t in tables]

        def evaluate(p, cell_inx=None, _t=curves):
            index = get_reg_map(p, reg.ROCKNUM, reg.ROCKINX, cell_inx=cell_inx)
            return interp_reg(_t, p, index)
        return evaluate

    f["pvMultR"] = _dispatch(1)
    f["tranMultR"] = _dispatch(2)
    return f


assignROCKTAB = assign_rocktab


if __name__ == "__main__":
    from pymrst.modules.ad_props.utils.handle_regions import Regions

    #         p       pvMult  tranMult
    TABLE = np.array([[1.0e6, 0.98, 0.90],
                      [2.0e7, 1.00, 1.00],
                      [5.0e7, 1.05, 1.20]])

    def test_pore_volume_multiplier_is_read_from_column_one() -> None:
        f = assign_rocktab({}, [TABLE], Regions())
        assert np.allclose(f["pvMultR"](np.array([2.0e7])), 1.00)
        print("ok test_pore_volume_multiplier_is_read_from_column_one")

    def test_transmissibility_multiplier_is_read_from_column_two() -> None:
        f = assign_rocktab({}, [TABLE], Regions())
        assert np.allclose(f["tranMultR"](np.array([5.0e7])), 1.20)
        print("ok test_transmissibility_multiplier_is_read_from_column_two")

    def test_the_two_multipliers_are_independent() -> None:
        f = assign_rocktab({}, [TABLE], Regions())
        probe = np.array([1.0e6])
        assert not np.allclose(f["pvMultR"](probe), f["tranMultR"](probe))
        print("ok test_the_two_multipliers_are_independent")

    def test_a_bare_table_is_accepted() -> None:
        """MRST wraps a non-cell argument; a single-region deck sends one."""
        f = assign_rocktab({}, TABLE, Regions())
        assert np.allclose(f["pvMultR"](np.array([2.0e7])), 1.00)
        print("ok test_a_bare_table_is_accepted")

    def test_it_partitions_on_rocknum_not_pvtnum() -> None:
        second = TABLE.copy()
        second[:, 1] *= 0.5
        reg = Regions(ROCKNUM=np.array([0, 1]),
                      ROCKINX=[np.array([0]), np.array([1])],
                      PVTNUM=np.array([1, 0]),
                      PVTINX=[np.array([1]), np.array([0])])
        f = assign_rocktab({}, [TABLE, second], reg)
        got = f["pvMultR"](np.full(2, 2.0e7))
        assert np.allclose(got, [1.00, 0.50]), got
        print("ok test_it_partitions_on_rocknum_not_pvtnum")

    def test_it_replaces_rocks_pv_multiplier() -> None:
        from pymrst.modules.ad_props.props.assign_rock import assign_rock
        f = assign_rock({}, [[2.0e7, 1.0e-9]], Regions())
        assert callable(f["pvMultR"])
        f = assign_rocktab(f, [TABLE], Regions())
        assert np.allclose(f["pvMultR"](np.array([5.0e7])), 1.05)
        print("ok test_it_replaces_rocks_pv_multiplier")

    test_pore_volume_multiplier_is_read_from_column_one()
    test_transmissibility_multiplier_is_read_from_column_two()
    test_the_two_multipliers_are_independent()
    test_a_bare_table_is_accepted()
    test_it_partitions_on_rocknum_not_pvtnum()
    test_it_replaces_rocks_pv_multiplier()
    print("\nAll tests passed!")
