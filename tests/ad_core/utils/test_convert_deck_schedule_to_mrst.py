from types import SimpleNamespace

import numpy as np
import pytest

from pymrst.modules.ad_core.utils import convert_deck_schedule_to_mrst
from pymrst.gridprocessing import cart_grid, compute_geometry
from pymrst.params import make_rock
from pymrst.params.wells_and_bc import Well
from pymrst.utils.units import DARCY, MILLI


def _model(phases=("water", "oil"), G=None, rock=None):
    return SimpleNamespace(
        fluid=SimpleNamespace(phases=phases), G=G, rock=rock, gravity=9.80665
    )


def _well(name, compi, cell=0, status=True):
    return Well(
        cells=np.array([cell]), type="rate", val=1.0, r=np.array([0.1]),
        dir="z", WI=np.array([2.0]), dZ=np.array([0.0]), name=name,
        compi=np.asarray(compi, float), refDepth=1000.0, sign=1,
        status=status,
    )


def test_steps_phase_mapping_and_consistent_well_order():
    deck = {
        "step": {"val": [10.0, 20.0, 30.0], "control": [1, 2, 2]},
        "control": [
            {"W": [_well("A", [0.2, 0.3, 0.5])]},
            {"W": [_well("B", [0.6, 0.4, 0.0])]},
        ],
    }
    schedule = convert_deck_schedule_to_mrst(_model(), deck)
    assert np.array_equal(schedule.step_control, [0, 1, 1])
    assert [w.name for w in schedule.controls[0].W] == ["A", "B"]
    assert [w.name for w in schedule.controls[1].W] == ["A", "B"]
    assert schedule.controls[0].W[1].status is False
    assert schedule.controls[1].W[0].status is False
    assert np.allclose(schedule.controls[0].W[0].compi, [0.4, 0.6])
    assert np.allclose(schedule.controls[1].W[1].compi, [0.6, 0.4])


def test_nested_deck_step_limit_and_dictionary_well():
    deck = {"SCHEDULE": {
        "step": {"val": [1.0, 2.0], "control": [1, 1]},
        "control": [{"W": [{
            "cells": [2], "name": "P", "type": "bhp", "val": 2.0e7,
            "WI": [3.0], "r": [0.1], "dZ": [4.0], "compi": [1, 0, 0],
        }]}],
    }}
    schedule = convert_deck_schedule_to_mrst(_model(), deck, step_limit=1)
    assert schedule.n_steps == 1
    assert schedule.controls[0].W[0].cells.tolist() == [2]
    assert schedule.controls[0].W[0].name == "P"


def test_raw_records_require_grid_and_rock():
    deck = {
        "step": {"val": [1.0], "control": [1]},
        "control": [{"WELSPECS": [["P", "G", 1, 1]], "COMPDAT": []}],
    }
    with pytest.raises(ValueError, match="model.G and model.rock"):
        convert_deck_schedule_to_mrst(_model(), deck)


def test_raw_completions_and_controls_persist_across_periods():
    grid = compute_geometry(cart_grid([2, 1, 2], [20.0, 10.0, 20.0]))
    rock = make_rock(grid, 100 * MILLI * DARCY, 0.25)
    model = _model(("water", "oil", "gas"), grid, rock)
    welspecs = [
        ["I", "FIELD", 1, 1, 0.0, "GAS", 0, "STD", "SHUT", "YES", 0, "SEG", 0],
        ["P", "FIELD", 2, 1, 0.0, "OIL", 0, "STD", "SHUT", "YES", 0, "SEG", 0],
    ]
    compdat = [
        ["I", 1, 1, 1, 2, "OPEN", 1, 5.0, 0.2, -1, 0, "Default", "Z", -1],
        ["P", 2, 1, 1, 2, "OPEN", 1, 6.0, 0.2, -1, 0, "Default", "Z", -1],
    ]
    deck = {
        "step": {"val": [10.0, 20.0], "control": [1, 2]},
        "control": [
            {
                "WELSPECS": welspecs,
                "COMPDAT": compdat,
                "WCONINJE": [["I", "GAS", "OPEN", "RATE", 100.0, 0, 300e5, 0, 0, 0, 0, 0, 0, 0]],
                "WCONPROD": [["P", "OPEN", "ORAT", 200.0, 0, 0, 0, 0, 100e5, 0, 0, 0]],
            },
            {
                "WCONINJE": [["I", "GAS", "SHUT", "RATE", 50.0, 0, 300e5, 0, 0, 0, 0, 0, 0, 0]],
                "WCONPROD": [["P", "OPEN", "BHP", 0, 0, 0, 0, 0, 150e5, 0, 0, 0]],
            },
        ],
    }
    schedule = convert_deck_schedule_to_mrst(model, deck)
    first = {well.name: well for well in schedule.controls[0].W}
    second = {well.name: well for well in schedule.controls[1].W}
    assert first["I"].cells.tolist() == [0, 2]
    assert first["P"].cells.tolist() == [1, 3]
    assert first["I"].type == "rate" and first["I"].val == 100.0
    assert first["P"].type == "orat" and first["P"].val == -200.0
    assert np.array_equal(first["I"].compi, [0.0, 0.0, 1.0])
    assert second["I"].status is False and second["I"].cells.tolist() == [0, 2]
    assert second["I"].val == 100.0 and np.all(second["I"].WI == 0.0)
    assert second["P"].type == "bhp" and second["P"].val == 150e5
