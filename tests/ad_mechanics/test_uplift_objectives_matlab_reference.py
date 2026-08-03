from pathlib import Path

import numpy as np
from scipy.io import loadmat

from pymrst.modules.ad_mechanics.examples.utils.compute_uplift_for_state import compute_uplift_for_state
from pymrst.modules.ad_mechanics.examples.utils.obj_uplift import obj_uplift
from pymrst.modules.ad_mechanics.examples.utils.obj_uplift_aver import obj_uplift_aver
from pymrst.utils.adi import init_variables_adi


FIXTURE = (Path(__file__).resolve().parents[2] / "scripts" / "matlab" /
           "ad_mechanics" / "ref_uplift_objectives.mat")


def _model():
    grid = type("Grid", (), {"griddim": 2, "nodes": type("Nodes", (), {"num": 3})()})()
    op = type("Operators", (), {
        "is_dir": np.array([True, True, True, False, False, False]),
        "u_dir": np.zeros(6),
    })()
    mech = type("Mechanics", (), {"operators": op})()
    return type("Model", (), {"G": grid, "mech": mech})()


def test_uplift_values_objectives_and_derivative_match_mrst():
    ref = loadmat(FIXTURE, squeeze_me=True, struct_as_record=False)
    model = _model()
    state1 = {"xd": np.array([2.0, 3.0, 4.0])}
    state2 = {"xd": np.array([-3.0, 5.0, 6.0])}
    schedule = {"step": {"val": np.array([4.0, 5.0])}}
    assert compute_uplift_for_state(model, state1, 1) == float(ref["uplift"])
    (xd,) = init_variables_adi(state1["xd"])
    uplift_ad = compute_uplift_for_state(model, {"xd": xd}, 1, compute_partials=True)
    np.testing.assert_allclose(uplift_ad.jac[0].toarray(), np.atleast_2d(ref["uplift_jac"]))
    np.testing.assert_allclose(obj_uplift(model, [state1, state2], schedule, 1), np.atleast_1d(ref["linear_values"]))
    np.testing.assert_allclose(
        obj_uplift_aver(model, [state1, state2], schedule, 1,
                       exponent=2, normalization_constant=0.5),
        np.atleast_1d(ref["power_values"]), rtol=1e-14, atol=1e-25,
    )
