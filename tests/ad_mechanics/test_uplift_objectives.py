import numpy as np
import pytest

from pymrst.modules.ad_mechanics.examples.utils.compute_uplift_for_state import compute_uplift_for_state
from pymrst.modules.ad_mechanics.examples.utils.obj_uplift_aver import obj_uplift_aver
from pymrst.utils.adi import init_variables_adi, value


def model_2d():
    grid = type("Grid", (), {"griddim": 2, "nodes": type("Nodes", (), {"num": 3})()})()
    operators = type(
        "Operators", (),
        {"is_dir": np.array([True, True, True, False, False, False]), "u_dir": np.array([0.0, -0.1, 0.0, 0.0, 0.0, 0.0])},
    )()
    mech = type("Mechanics", (), {"operators": operators})()
    return type("Model", (), {"G": grid, "mech": mech})()


def test_uplift_maps_full_node_dof_to_reduced_xd():
    model = model_2d()
    state = {"xd": np.array([2.0, 3.0, 4.0])}
    assert compute_uplift_for_state(model, state, 1) == 2.0
    assert compute_uplift_for_state(model, state, 0) == pytest.approx(-0.1)


def test_uplift_preserves_ad_derivative():
    model = model_2d()
    (xd,) = init_variables_adi(np.array([2.0, 3.0, 4.0]))
    uplift = compute_uplift_for_state(model, {"xd": xd}, 1, compute_partials=True)
    assert value(uplift) == pytest.approx([2.0])
    assert np.allclose(uplift.jac[0].toarray(), [[1.0, 0.0, 0.0]])


def test_average_objective_matches_mrst_positive_power_and_timestep_weighting():
    model = model_2d()
    states = [{"xd": np.array([2.0, 0.0, 0.0])}, {"xd": np.array([-3.0, 0.0, 0.0])}]
    schedule = {"step": {"val": np.array([4.0, 5.0])}}
    result = obj_uplift_aver(
        model, states, schedule, 1, exponent=2, normalization_constant=0.5
    )
    assert result[0] == pytest.approx(4.0)
    assert result[1] == pytest.approx(0.0, abs=1e-25)


def test_average_objective_single_step_contract():
    model = model_2d()
    state = {"xd": np.array([2.0, 0.0, 0.0])}
    result = obj_uplift_aver(
        model, [state], {"step": {"val": [4.0, 5.0]}}, 1,
        compute_all_steps=False, t_step=2, state=state,
    )
    assert result[0] == pytest.approx(10.0)
