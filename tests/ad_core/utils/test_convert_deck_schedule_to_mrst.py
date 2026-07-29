from types import SimpleNamespace

import numpy as np
import pytest

from pymrst.modules.ad_core.utils import convert_deck_schedule_to_mrst
from pymrst.params.wells_and_bc import Well


def _model(phases=("water", "oil")):
    return SimpleNamespace(fluid=SimpleNamespace(phases=phases))


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


def test_raw_eclipse_well_keywords_are_explicitly_rejected():
    deck = {
        "step": {"val": [1.0], "control": [1]},
        "control": [{"WELSPECS": [["P", "G", 1, 1]], "COMPDAT": []}],
    }
    with pytest.raises(NotImplementedError, match="processWells"):
        convert_deck_schedule_to_mrst(_model(), deck)
