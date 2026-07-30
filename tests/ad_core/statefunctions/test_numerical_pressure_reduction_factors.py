"""Tests for MRST ``NumericalPressureReductionFactors``."""

import numpy as np

from pymrst.gridprocessing import cart_grid, compute_geometry
from pymrst.modules.ad_blackoil.models.generic_black_oil_model import (
    GenericBlackOilModel,
)
from pymrst.modules.ad_core.statefunctions.flowprops import (
    NumericalPressureReductionFactors,
)
from pymrst.modules.ad_core.statefunctions.state_function import State
from pymrst.modules.ad_props import init_simple_adi_fluid
from pymrst.params import make_rock
from pymrst.utils.adi import init_variables_adi, value
from pymrst.utils.units import BARSA, CENTI, DARCY, MILLI, POISE


def test_local_mass_jacobian_weights_cancel_nonpressure_derivative():
    pressure, saturation = init_variables_adi(
        np.array([10.0, 20.0]), np.array([0.2, 0.7])
    )
    mass = [
        2.0 * pressure + 3.0 * saturation,
        5.0 * pressure + 7.0 * saturation,
    ]
    weights = NumericalPressureReductionFactors.get_weights(
        mass, [np.zeros(2), np.zeros(2)], 1.0, ncell=2, ncomp=2
    )
    expected = np.linalg.solve(
        np.array([[2.0, 5.0], [3.0, 7.0]]),
        np.array([1.0 / BARSA, 0.0]),
    )
    assert np.allclose(weights, expected[None, :])

    combined = weights[:, 0] * mass[0] + weights[:, 1] * mass[1]
    assert np.allclose(
        combined.jac[0].toarray(), np.eye(2) / BARSA
    )
    assert np.allclose(combined.jac[1].toarray(), 0.0)


def test_numeric_mass_fallback_and_weight_pressure_derivative():
    numeric = NumericalPressureReductionFactors.get_weights(
        [np.ones(2), np.ones(2)],
        [np.zeros(2), np.zeros(2)],
        1.0,
        ncell=2,
        ncomp=2,
    )
    assert np.array_equal(numeric, np.ones((2, 2)))

    derivative = NumericalPressureReductionFactors.get_weight_derivatives(
        weights=np.array([[2.0, 1.0], [4.0, 2.0]]),
        previous_weights=np.array([[1.0, 1.0], [2.0, 2.0]]),
        pressure=np.array([110.0, 200.0]),
        previous_pressure=np.array([100.0, 200.0]),
        mass=[np.ones(2), np.ones(2)],
    )
    assert np.isfinite(derivative).all()
    assert np.array_equal(derivative[1], [0.0, 0.0])


def test_evaluate_on_real_model_returns_ad_pressure_weights():
    grid = compute_geometry(cart_grid([3, 1, 1], [30.0, 10.0, 10.0]))
    rock = make_rock(grid, 100 * MILLI * DARCY, 0.25)
    fluid = init_simple_adi_fluid(
        mu=np.array([1.0, 5.0]) * CENTI * POISE,
        rho=[1000.0, 800.0],
        n=[2, 2],
        c=[1e-6 / BARSA, 1e-4 / BARSA],
        p_ref=200 * BARSA,
    )
    model = GenericBlackOilModel(grid, rock, fluid, [])
    base = State(
        pressure=np.full(3, 200 * BARSA),
        sW=np.array([0.2, 0.5, 0.8]),
        qWs=np.zeros(0),
        qOs=np.zeros(0),
        bhp=np.zeros(0),
    )
    state = model.init_state_ad(base)
    mass = model.get_props(state, "ComponentTotalMass")
    state["reductionFactorProps"] = {
        "dt": 1.0,
        "mass0": [np.zeros(3), np.zeros(3)],
        "weights": np.ones((3, 2)),
        "pressure": np.full(3, 199 * BARSA),
    }
    weights = NumericalPressureReductionFactors(model).evaluate_on_domain(
        model, state
    )
    assert len(weights) == 2
    for component_weights in weights:
        assert np.isfinite(value(component_weights)).all()
        assert np.isfinite(component_weights.jac[0].toarray()).all()
