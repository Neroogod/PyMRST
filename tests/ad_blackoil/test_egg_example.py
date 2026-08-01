import numpy as np

from pymrst.modules.ad_blackoil.examples.egg import (
    compare_egg_production_history, setup_egg, run_egg_example,
)
from pymrst.modules.deckformat import read_egg_deck
from pymrst.utils.units import DAY


def test_egg_dataset_properties_realizations_and_schedule():
    deck = read_egg_deck()
    assert deck["cartDims"] == (60, 60, 7)
    assert deck["actnum"].sum() == 18553
    assert deck["perm"].shape == (25200, 3)
    assert deck["tsteps"].size == 123
    assert np.isclose(deck["tsteps"].sum()/DAY, 3600)
    realization = read_egg_deck(realization=1)
    assert not np.allclose(realization["perm"], deck["perm"], rtol=1e-10, atol=0)


def test_egg_setup_maps_all_wells_and_active_cells():
    _, model, schedule, state = setup_egg()
    assert model.G.cells.num == 18553
    assert len(model.wells) == 12
    assert sum(w.sign > 0 for w in model.wells) == 8
    assert sum(w.sign < 0 for w in model.wells) == 4
    assert all(len(w.cells) == 7 for w in model.wells)
    assert schedule.n_steps == 123
    assert np.allclose(state["sW"], 0.1)


def test_egg_first_report_step_converges_and_reports_production():
    result = run_egg_example(1, verbose=False)
    assert result.report.converged
    assert result.history["qOs"].shape == (1, 12)
    assert np.all(result.history["qOs"][0, 8:] < 0)
    assert np.all(result.history["qWs"][0, :8] > 0)
    errors = compare_egg_production_history(
        result, "examples/validation/ref_ad_blackoil_egg.mat"
    )
    assert errors["qOs"] < 0.012
