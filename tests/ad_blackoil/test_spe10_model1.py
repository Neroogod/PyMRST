import numpy as np
import pytest

from pymrst.modules.ad_blackoil.examples.spe10 import (
    run_tenth_csp_model_i_ad, setup_tenth_csp_model_i_ad,
)
from pymrst.modules.deckformat import load_spe10_model1
from pymrst.utils.units import DAY


def test_spe10_model1_loader_and_layer_selection():
    full = load_spe10_model1()
    assert full["cartDims"] == (100, 1, 20)
    assert full["perm"].shape == (2000, 3)
    assert np.all(full["poro"] == 0.2)
    subset = load_spe10_model1(layers=[1, 20])
    assert subset["perm"].shape == (200, 3)
    assert np.array_equal(subset["indexMap"][:3], [0, 1, 2])
    assert np.array_equal(subset["indexMap"][-3:], [1997, 1998, 1999])
    with pytest.raises(ValueError, match="1..20"):
        load_spe10_model1(layers=[0])


def test_spe10_model1_complete_initialization():
    data, model, schedule, state = setup_tenth_csp_model_i_ad()
    assert model.G.cells.num == 2000
    assert len(model.wells) == 2
    assert all(w.cells.size == 20 for w in model.wells)
    assert schedule.n_steps == 131
    assert np.isclose(schedule.total_time/DAY, 3650.0)
    assert np.all(state["sW"] == 1.0)
    assert np.array_equal(model.G.cells.indexMap, data["indexMap"])


def test_spe10_model1_first_report_converges():
    result = run_tenth_csp_model_i_ad(1, verbose=False)
    assert result.report.converged
    assert result.history["sG"].max() > 0.0
    assert np.isclose(result.history["bhp"][0, 1], 95*6894.757293168361)
