import numpy as np
import pytest

from pymrst.modules.ad_mechanics.examples.mandel import mandel_pressure, run_mandel
from pymrst.modules.ad_mechanics.examples.terzaghi import run_terzaghi, terzaghi_pressure
from pymrst.modules.ad_mechanics.examples.utils.obj_uplift import obj_uplift


def test_mandel_example_validates_time_sequence_before_expensive_setup():
    with pytest.raises(ValueError, match="strictly increasing"):
        run_mandel([0.1, 0.1])


def test_mandel_analytic_profiles_have_expected_shape():
    out = mandel_pressure(np.array([0.0, 0.5, 1.0]), np.array([0.01, 0.1]), 0.2, 0.44)
    assert out.shape == (2, 3)
    assert np.allclose(out[:, -1], 0.0, atol=1e-12)


def test_mandel_complete_time_history_and_error_schema():
    result = run_mandel([0.05, 0.2], nx=12, nz=3, substeps=4)
    assert result["schema_version"] == 1
    assert result["pressure_profiles"].shape == result["analytic_profiles"].shape
    assert result["pressure_profiles"].shape[0] == 2
    assert result["max_errors"].shape == (2,)
    assert np.max(result["max_errors"]) < 0.05
    assert len(result["states"]) == 2


def test_obj_uplift_weights_each_state_by_its_timestep():
    model = type("Model", (), {"G": type("Grid", (), {"griddim": 3})()})()
    states = [
        {"u": np.array([0.0, 0.0, -2.0, 0.0, 0.0, -3.0])},
        {"u": np.array([0.0, 0.0, -4.0, 0.0, 0.0, -5.0])},
    ]
    schedule = {"step": {"val": np.array([2.0, 3.0])}}
    assert obj_uplift(model, states, schedule, 1) == pytest.approx([6.0, 15.0])
    assert obj_uplift(
        model, states, schedule, 1, compute_all_steps=False, t_step=1, state=states[1]
    ) == pytest.approx([15.0])


def test_terzaghi_example_has_complete_time_history_and_schema():
    result = run_terzaghi(layers=120, substeps=40)
    assert result["pressure_profiles_pa"].shape == (6, 120)
    assert result["analytic_pressure_profiles_pa"].shape == (6, 120)
    assert result["top_displacement_m"].shape == (6,)
    assert result["schema_version"] == 1
    assert np.max(result["pressure_relative_max_errors"][1:]) < 0.03
    assert np.max(result["settlement_relative_errors"][1:]) < 0.03
    assert np.all(np.diff(result["top_displacement_m"]) > 0)
    assert np.allclose(terzaghi_pressure([0.0], [0.1]), 0.0, atol=1e-14)
