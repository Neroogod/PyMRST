from pathlib import Path

import numpy as np
import pytest

from pymrst.modules.ad_core.initialization import init_eclipse_problem_ad
from pymrst.modules.ad_core.simulators.nonlinear_solver import NonLinearSolver
from pymrst.modules.ad_core.utils.schedule import Schedule


SPE1 = (
    Path(__file__).resolve().parents[4]
    / "mrst-2024a-Geomech/examples/data/SPE1/BENCH_SPE1.DATA"
)


@pytest.mark.skipif(not SPE1.exists(), reason="local MRST SPE1 deck is unavailable")
def test_spe1_path_builds_complete_problem():
    state, model, schedule, nonlinear = init_eclipse_problem_ad(SPE1)
    assert state["pressure"].shape == (model.G.cells.num,)
    assert np.all(state["pressure"] > 0)
    assert isinstance(schedule, Schedule)
    assert schedule.n_steps == 120
    assert np.allclose(schedule.step_val, model.input_data["tsteps"])
    assert isinstance(nonlinear, NonLinearSolver)
    assert nonlinear.max_iterations == 12


@pytest.mark.skipif(not SPE1.exists(), reason="local MRST SPE1 deck is unavailable")
def test_model_override_and_output_switches():
    state, model, _, _ = init_eclipse_problem_ad(SPE1, get_schedule=False)
    state2, model2, schedule2, nonlinear2 = init_eclipse_problem_ad(
        {"pressure": state["pressure"], "swat": state["sW"],
         "sgas": state["sG"], "rs": state["rs"]},
        model=model, get_initial_state=False, get_schedule=False,
        timestep_strategy="none", use_cpr=False,
    )
    assert state2 is None and schedule2 is None and model2 is model
    assert nonlinear2.timestep_selector is None


@pytest.mark.skipif(not SPE1.exists(), reason="local MRST SPE1 deck is unavailable")
def test_model_override_converts_processed_deck_schedule():
    _, model, _, _ = init_eclipse_problem_ad(SPE1, get_schedule=False)
    deck = {
        "pressure": np.full(model.G.cells.num, 2.0e7),
        "swat": np.full(model.G.cells.num, 0.2),
        "sgas": np.zeros(model.G.cells.num),
        "rs": np.zeros(model.G.cells.num),
        "SCHEDULE": {
            "step": {"val": [1.0, 2.0], "control": [1, 1]},
            "control": [{"W": []}],
        },
    }
    _, _, schedule, _ = init_eclipse_problem_ad(
        deck, model=model, get_initial_state=False
    )
    assert schedule.n_steps == 2
    assert np.array_equal(schedule.step_control, [0, 0])
