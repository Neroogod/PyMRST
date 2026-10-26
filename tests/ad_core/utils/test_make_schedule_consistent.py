import numpy as np

from pymrst.modules.ad_core.utils import (
    Control,
    Schedule,
    make_schedule_consistent_ad,
)
from pymrst.params.wells_and_bc import Well


def _well(name, cells, wi, *, val=10.0, status=True, cstatus=None):
    cells = np.asarray(cells, dtype=np.int64)
    return Well(
        cells=cells, type="rate", val=val,
        r=np.full(cells.size, 0.1), dir="z", WI=np.asarray(wi, float),
        dZ=np.arange(cells.size, dtype=float), name=name,
        compi=np.array([1.0, 0.0]), refDepth=0.0, sign=1,
        status=status, cstatus=cstatus,
    )


def test_union_of_wells_and_perforations_is_stable():
    schedule = Schedule(
        [1.0, 2.0], [0, 1],
        [
            Control(W=[_well("A", [0, 2], [5.0, 7.0])]),
            Control(W=[
                _well("A", [2, 3], [8.0, 9.0]),
                _well("B", [1], [4.0], val=20.0),
            ]),
        ],
    )
    result = make_schedule_consistent_ad(schedule)
    assert [w.name for w in result.controls[0].W] == ["A", "B"]
    assert [w.name for w in result.controls[1].W] == ["A", "B"]
    assert np.array_equal(result.controls[0].W[0].cells, [0, 2, 3])
    assert np.array_equal(result.controls[0].W[0].WI, [5.0, 7.0, 0.0])
    assert np.array_equal(result.controls[1].W[0].WI, [0.0, 8.0, 9.0])
    assert result.controls[0].W[1].status is False
    assert np.all(result.controls[0].W[1].WI == 0.0)
    assert result.controls[0].W[1].val == 20.0


def test_shut_well_keeps_first_active_control_but_closes_connections():
    schedule = Schedule(
        [1.0, 1.0], [0, 1],
        [
            Control(W=[_well("A", [0], [5.0], val=100.0)]),
            Control(W=[_well("A", [0], [5.0], val=50.0, status=False)]),
        ],
    )
    result = make_schedule_consistent_ad(schedule)
    shut = result.controls[1].W[0]
    assert shut.status is False
    assert shut.val == 100.0
    assert np.all(shut.WI == 0.0)
    assert np.all(shut.cstatus == 0)
