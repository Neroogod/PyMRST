from pathlib import Path
import numpy as np

from pymrst.modules.ad_blackoil.examples.spe9 import setup_spe9
from pymrst.modules.deckformat import read_spe9_deck
from pymrst.utils.units import DAY

ROOT = Path(__file__).resolve().parents[2]
DECK = ROOT.parent/"mrst-2024a-Geomech/examples/data/SPE9/BENCH_SPE9.DATA"


def test_spe9_deck_grid_pvt_initial_and_schedule():
    deck = read_spe9_deck(DECK)
    assert deck["cartDims"] == (24, 25, 15)
    assert deck["pressure"].size == 9000
    assert deck["perm"].shape == (9000, 3)
    np.testing.assert_allclose(deck["perm"][:, 2], 0.01*deck["perm"][:, 0])
    assert len(deck["pvto"]) == 11 and deck["pvdg"].shape == (11, 3)
    assert deck["tsteps"].size == 35
    assert np.isclose(deck["tsteps"].sum()/DAY, 900)
    assert list(deck["control_switch_steps"]) == [0, 16, 19]
    assert all(len(c["WCONPROD"]) == 25 for c in deck["controls"])


def test_spe9_full_setup_has_all_wells_and_control_changes():
    _, model, schedule, state = setup_spe9()
    assert model.G.cells.num == 9000
    assert len(model.wells) == 26
    assert state["pressure"].size == 9000
    assert np.all(state["status"] == 1)
    assert [len(c.W) for c in schedule.controls] == [26, 26, 26]
    prod = next(i for i, w in enumerate(schedule.controls[0].W) if w.name == "PROD2")
    rates = [-c.W[prod].val for c in schedule.controls]
    assert rates[0] > rates[1] and np.isclose(rates[0], rates[2])
    injector = schedule.controls[0].W[0]
    assert injector.name == "INJE1" and injector.type == "rate" and injector.sign == 1
