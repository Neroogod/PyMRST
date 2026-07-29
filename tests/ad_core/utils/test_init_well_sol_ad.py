from types import SimpleNamespace

import numpy as np

from pymrst.modules.ad_core.utils import init_well_sol_ad
from pymrst.params.wells_and_bc import Well
from pymrst.utils.units import BARSA


def _well(name, cell, kind, value, sign, compi):
    return Well(
        cells=np.array([cell]), type=kind, val=value, r=np.array([0.1]),
        dir="z", WI=np.array([1.0]), dZ=np.array([0.0]), name=name,
        compi=np.asarray(compi, float), refDepth=0.0, sign=sign,
        cstatus=np.array([True]),
    )


def test_default_bhp_guess_and_schedule_rate_assignment():
    model = SimpleNamespace(fluid=SimpleNamespace(phases=("water", "oil", "gas")))
    wells = [
        _well("I", 0, "rate", 100.0, 1, [0.0, 0.0, 1.0]),
        _well("P", 1, "orat", -50.0, -1, [0.0, 1.0, 0.0]),
    ]
    rows = init_well_sol_ad(
        wells, model, {"pressure": np.array([200.0, 180.0]) * BARSA}
    )
    assert rows[0].bhp == 205.0 * BARSA
    assert rows[1].bhp == 175.0 * BARSA
    assert rows[0].qGs == 100.0
    assert rows[0].qWs == 0.0 and rows[0].qOs == 0.0
    assert rows[1].qOs == -50.0
    assert rows[0].cqs.shape == (1, 3)


def test_bhp_control_overrides_directional_guess():
    model = SimpleNamespace(fluid=SimpleNamespace(phases=("water", "oil")))
    well = _well("P", 0, "bhp", 150.0 * BARSA, -1, [0.0, 1.0])
    row = init_well_sol_ad(
        [well], model, {"pressure": np.array([200.0 * BARSA])}
    )[0]
    assert row.bhp == 150.0 * BARSA
    assert row.pressure == row.bhp


def test_explicit_rows_take_precedence_but_receive_new_schedule_control():
    model = SimpleNamespace(fluid=SimpleNamespace(phases=("water", "oil")))
    well = _well("I", 0, "rate", 25.0, 1, [1.0, 0.0])
    existing = [SimpleNamespace(bhp=123.0, qWs=0.0, qOs=0.0)]
    row = init_well_sol_ad(
        [well], model, {"pressure": np.array([1.0])}, existing
    )[0]
    assert row.bhp == 123.0
    assert row.qWs == 25.0 and row.type == "rate"
