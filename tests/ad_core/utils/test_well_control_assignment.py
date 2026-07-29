from types import SimpleNamespace

import numpy as np

from pymrst.modules.ad_core.utils import (
    assign_well_values_from_control,
    set_well_sign,
)
from pymrst.params.wells_and_bc import Well


def _well(name, kind, value, sign, compi):
    return Well(
        cells=np.array([0]), type=kind, val=value, r=np.array([0.1]),
        dir="z", WI=np.array([1.0]), dZ=np.array([0.0]), name=name,
        compi=np.asarray(compi, float), refDepth=0.0, sign=sign,
    )


def test_missing_sign_is_inferred_without_overwriting_existing_sign():
    wells = [
        _well("P", "rate", -10.0, None, [0, 1, 0]),
        _well("I", "rate", 10.0, None, [1, 0, 0]),
        _well("B", "bhp", 100.0, None, [0, 1, 0]),
        _well("KEEP", "bhp", 100.0, -1, [0, 1, 0]),
    ]
    out = set_well_sign(wells)
    assert [well.sign for well in out] == [-1, 1, 0, -1]


def test_controls_assign_bhp_injection_and_phase_rates():
    model = SimpleNamespace(fluid=SimpleNamespace(phases=("water", "oil", "gas")))
    wells = [
        _well("I", "rate", 100.0, 1, [0.2, 0.3, 0.5]),
        _well("P", "rate", -80.0, -1, [0, 1, 0]),
        _well("B", "bhp", 150.0, -1, [0, 1, 0]),
        _well("O", "orat", -40.0, -1, [0, 1, 0]),
    ]
    rows = [
        SimpleNamespace(status=True, type=w.type, val=w.val, sign=w.sign,
                        bhp=0.0, qWs=1.0, qOs=2.0, qGs=3.0)
        for w in wells
    ]
    out = assign_well_values_from_control(model, rows, wells)
    assert (out[0].qWs, out[0].qOs, out[0].qGs) == (20.0, 30.0, 50.0)
    assert (out[1].qWs, out[1].qOs, out[1].qGs) == (1.0, 2.0, 3.0)
    assert out[2].bhp == 150.0
    assert out[3].qOs == -40.0
