import numpy as np

from pymrst.gridprocessing import cart_grid, compute_geometry
from pymrst.modules.ad_blackoil.models.two_phase_oil_water_model import (
    TwoPhaseOilWaterModelAD,
)
from pymrst.modules.ad_core.simulators.compute_sensitivities_adjoint_ad import (
    AdjointParameter,
    compute_sensitivities_adjoint_ad,
)
from pymrst.modules.optimization.adjoint import (
    evaluate_objective,
    npv_objective,
    simulate_forward,
)
from pymrst.modules.ad_props import init_simple_adi_fluid
from pymrst.params import add_well, make_rock
from pymrst.utils.units import BARSA, DAY


def _case():
    grid = compute_geometry(cart_grid([4, 3, 1], [120.0, 90.0, 10.0]))
    rock = make_rock(grid, 1e-13, 0.2)
    fluid = init_simple_adi_fluid(
        mu=[1e-3, 5e-3], rho=[1000.0, 800.0], n=[2, 2],
        c=[1e-10, 1e-9], p_ref=200 * BARSA,
    )
    wells = add_well(
        None, grid, rock, np.array([0]), well_type="rate", val=40 / DAY,
        compi=np.array([1.0, 0.0]), name="I",
    )
    wells = add_well(
        wells, grid, rock, np.array([grid.cells.num - 1]), well_type="bhp",
        val=190 * BARSA, compi=np.array([0.0, 1.0]), name="P",
    )
    model = TwoPhaseOilWaterModelAD(grid, rock, fluid, wells)
    state0 = model.initial_state(200 * BARSA, 0.2)
    dts = [15 * DAY, 20 * DAY]
    controls = np.array([[40 / DAY, 190 * BARSA],
                         [40 / DAY, 190 * BARSA]])
    return model, state0, dts, controls


def test_shared_control_parameter_matches_centered_difference():
    model, state0, dts, controls = _case()
    objective = npv_objective()
    setup = {"model": model, "state0": state0, "dts": dts,
             "controls": controls}
    params = [
        AdjointParameter("inj_rate", "rate", subset=[0],
                         control_steps=[0, 1]),
        AdjointParameter("prod_bhp", "bhp", subset=[1],
                         control_steps=[0, 1]),
    ]
    sens = compute_sensitivities_adjoint_ad(setup, None, params, objective)

    for parameter, well in zip(params, [0, 1]):
        h = 1e-6 * max(abs(controls[0, well]), 1.0)
        up, down = controls.copy(), controls.copy()
        up[:, well] += h
        down[:, well] -= h
        ju = evaluate_objective(
            model, objective, simulate_forward(model, state0, dts, up),
            dts, up,
        )
        jd = evaluate_objective(
            model, objective, simulate_forward(model, state0, dts, down),
            dts, down,
        )
        fd = (ju - jd) / (2 * h)
        assert np.isclose(sens[parameter.name][0], fd, rtol=2e-5)


def test_non_schedule_parameter_is_explicitly_rejected():
    model, state0, dts, controls = _case()
    parameter = AdjointParameter(
        "permx", "none", belongs_to="model", subset=[0]
    )
    setup = {"model": model, "state0": state0, "dts": dts,
             "controls": controls}
    try:
        compute_sensitivities_adjoint_ad(
            setup, None, [parameter], npv_objective()
        )
    except NotImplementedError:
        return
    raise AssertionError("model parameter seeding must not be silently approximated")
