"""The adjoint's sensitivity to the *initial state* (MRST `state0` parameters).

The backward sweep already ends holding the first step's multiplier, and one
more application of the coupling term it uses at every other step gives

    dJ/dx_0 = (dR_1/dx_0)^T lam_1

so this is the adjoint answer rather than an approximation of it, and it costs
one transpose-multiply against a matrix the sweep already builds - not another
simulation. That is the whole reason `state0` parameters are supportable while
rock parameters are not: rock needs the residual differentiated with respect to
porosity and permeability, which means seeding those as AD variables inside the
model's assembly.

The anchor is the finite-difference check. A gradient that is subtly wrong -
missing the coupling term, or transposed - still returns a plausible vector of
the right shape and length, and an optimiser fed one converges somewhere.
"""

from __future__ import annotations

import numpy as np
import pytest

from pymrst.gridprocessing import cart_grid, compute_geometry
from pymrst.modules.ad_blackoil.models.two_phase_oil_water_model import (
    TwoPhaseOilWaterModelAD,
)
from pymrst.modules.ad_core.simulators.compute_sensitivities_adjoint_ad import (
    AdjointParameter, compute_sensitivities_adjoint_ad,
)
from pymrst.modules.ad_props.init_simple_adi_fluid import init_simple_adi_fluid
from pymrst.modules.optimization.adjoint import (
    compute_gradient_adjoint, evaluate_objective, npv_objective,
    simulate_forward,
)
from pymrst.params.rock import make_rock
from pymrst.params.wells_and_bc import add_well
from pymrst.utils.units import BARSA, CENTI, DARCY, DAY, MILLI, POISE


def _setup(nx=4, ny=4, n_steps=3):
    g = compute_geometry(cart_grid([nx, ny, 1], [200.0, 200.0, 10.0]))
    rock = make_rock(g, 100*MILLI*DARCY, 0.2)
    fluid = init_simple_adi_fluid(
        mu=[1*CENTI*POISE, 5*CENTI*POISE], rho=[1000.0, 800.0],
        n=[2.0, 2.0], c=[1e-10, 1e-9], p_ref=200*BARSA)
    W = add_well(None, g, rock, np.array([0]), well_type="rate",
                 val=40.0/DAY, compi=np.array([1.0, 0.0]), name="I")
    W = add_well(W, g, rock, np.array([g.cells.num - 1]), well_type="bhp",
                 val=200*BARSA, compi=np.array([1.0, 0.0]), name="P")
    model = TwoPhaseOilWaterModelAD(g, rock, fluid, W)
    # The controls *vary between steps* deliberately. With identical rows,
    # `controls[0]` and `controls[-1]` are the same vector, and a bug that
    # evaluates the first step's coupling term under the last step's controls
    # cannot be detected at all.
    controls = np.array([[(40.0 + 10.0*n)/DAY, (200 + 5*n)*BARSA]
                         for n in range(n_steps)])
    return (model, model.initial_state(200*BARSA, 0.2),
            [20.0*DAY]*n_steps, controls, npv_objective())


# -- the anchor --------------------------------------------------------------

@pytest.mark.parametrize("index, step", [
    (0, 50.0), (5, 50.0), (11, 50.0),      # initial pressure, in Pa
    (18, 1e-5), (23, 1e-5),                # initial water saturation
])
def test_the_initial_state_gradient_matches_a_finite_difference(index, step):
    """The load-bearing check. A missing coupling term or a transposed matrix
    still returns a plausible vector of the right length."""
    model, state0, dts, controls, objective = _setup()
    result = compute_gradient_adjoint(model, state0, dts, controls, objective)
    n_cells = model.G.cells.num

    def perturbed(h):
        s = state0.copy()
        if index < n_cells:
            s.pressure = s.pressure.copy()
            s.pressure[index] += h
        else:
            s.sW = s.sW.copy()
            s.sW[index - n_cells] += h
        return evaluate_objective(
            model, objective, simulate_forward(model, s, dts, controls),
            dts, controls)

    finite = (perturbed(step) - perturbed(-step))/(2*step)
    adjoint = result.initial_state_gradient[index]
    assert adjoint == pytest.approx(finite, rel=1e-5)


def test_the_gradient_covers_the_whole_state_vector():
    """One entry per unknown: pressure and saturation per cell, plus the well
    unknowns. A gradient the length of the grid alone would silently omit the
    well block."""
    model, state0, dts, controls, objective = _setup()
    result = compute_gradient_adjoint(model, state0, dts, controls, objective)

    n_cells = model.G.cells.num
    assert result.initial_state_gradient.size > 2*n_cells
    assert np.all(np.isfinite(result.initial_state_gradient))


def test_the_control_gradient_is_unchanged():
    """The initial-state gradient is an addition, not a rewrite: the control
    gradients the sweep already produced must be untouched."""
    model, state0, dts, controls, objective = _setup()
    result = compute_gradient_adjoint(model, state0, dts, controls, objective)

    for n in range(len(dts)):
        for w in range(controls.shape[1]):
            # Relative to the control's own magnitude. An absolute step sized
            # for a pressure (1e7 Pa) is enormous next to a rate (5e-4 m3/s)
            # and simply makes the forward solve diverge.
            step = 1e-4*abs(controls[n, w])
            def shifted(h):
                u = controls.copy()
                u[n, w] += h
                return evaluate_objective(
                    model, objective,
                    simulate_forward(model, state0, dts, u), dts, u)
            finite = (shifted(step) - shifted(-step))/(2*step)
            assert result.gradient[n, w] == pytest.approx(finite, rel=1e-4)


# -- the parameter surface ---------------------------------------------------

def test_a_state0_parameter_returns_the_selected_entries():
    model, state0, dts, controls, objective = _setup()
    setup = {"model": model, "state0": state0, "dts": dts,
             "controls": controls}

    full = compute_gradient_adjoint(model, state0, dts, controls, objective)
    selected = compute_sensitivities_adjoint_ad(
        setup, None, [AdjointParameter(name="p0", control_type="",
                                       belongs_to="state0",
                                       subset=[0, 3, 7])],
        objective)

    np.testing.assert_allclose(
        selected["p0"], full.initial_state_gradient[[0, 3, 7]], rtol=1e-10)


def test_a_state0_parameter_without_a_subset_returns_everything():
    model, state0, dts, controls, objective = _setup()
    setup = {"model": model, "state0": state0, "dts": dts,
             "controls": controls}
    out = compute_sensitivities_adjoint_ad(
        setup, None, [AdjointParameter(name="all", control_type="",
                                       belongs_to="state0")],
        objective)

    full = compute_gradient_adjoint(model, state0, dts, controls, objective)
    np.testing.assert_allclose(out["all"], full.initial_state_gradient)


def test_a_subset_outside_the_state_vector_is_refused():
    model, state0, dts, controls, objective = _setup()
    setup = {"model": model, "state0": state0, "dts": dts,
             "controls": controls}
    with pytest.raises(IndexError, match="state vector"):
        compute_sensitivities_adjoint_ad(
            setup, None, [AdjointParameter(name="bad", control_type="",
                                           belongs_to="state0",
                                           subset=[99999])],
            objective)


def test_a_rock_parameter_says_what_is_missing_and_what_works():
    """The remaining gap, and the refusal must distinguish it from `state0` -
    otherwise a caller reads "adjoint parameters are unsupported" when half of
    them are."""
    model, state0, dts, controls, objective = _setup()
    setup = {"model": model, "state0": state0, "dts": dts,
             "controls": controls}
    with pytest.raises(NotImplementedError) as raised:
        compute_sensitivities_adjoint_ad(
            setup, None, [AdjointParameter(name="poro", control_type="",
                                           belongs_to="model")],
            objective)

    message = str(raised.value)
    assert "residual" in message, "say what is missing"
    # And *how* it is missing. "residual" alone survives a message that has
    # been reduced to "unsupported", which is the refusal this replaces.
    assert "AD variables" in message, "name what would have to change"
    assert "assemble" in message, "and where"
    assert "state0" in message, "and what is supported"


def test_schedule_parameters_still_work():
    """The path that already existed must be untouched by the new branch."""
    model, state0, dts, controls, objective = _setup()
    setup = {"model": model, "state0": state0, "dts": dts,
             "controls": controls}
    out = compute_sensitivities_adjoint_ad(
        setup, None,
        [AdjointParameter(name="rate", control_type="rate", subset=[0])],
        objective)

    full = compute_gradient_adjoint(model, state0, dts, controls, objective)
    assert out["rate"] == pytest.approx(np.sum(full.gradient[:, 0]))
