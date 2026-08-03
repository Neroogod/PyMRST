"""The adjoint gradient driver (F0.4 risk list).

An adjoint is the classic silently-plausible defect: a transposed Jacobian or a
dropped time-coupling term still returns a gradient of the right shape and
sign, and an optimizer using it simply converges somewhere else. Correctness
here is a mathematical property, not agreement with another code, so the ground
truth is a central finite difference of the objective.

That comparison is stronger here than the transport-Jacobian one that turned
out to prove less than it looked. There, residual and Jacobian shared a
formulation, so a consistent error moved both and every finite-difference test
passed. Here the finite difference re-runs the **full nonlinear forward
simulation** while the adjoint uses transposed Jacobians and a backward sweep;
an error in the sweep, the coupling term or the control Jacobian moves one and
not the other.
"""

from __future__ import annotations

import numpy as np
import pytest

from pymrst.gridprocessing import cart_grid, compute_geometry
from pymrst.modules.ad_blackoil.models.two_phase_oil_water_model import (
    TwoPhaseOilWaterModelAD,
)
from pymrst.modules.ad_props.init_simple_adi_fluid import init_simple_adi_fluid
from pymrst.modules.optimization.adjoint import (
    evaluate_objective, npv_objective, simulate_forward,
)
from pymrst.modules.optimization.utils.compute_gradient_adjoint_ad import (
    compute_gradient_adjoint,
)
from pymrst.params.rock import make_rock
from pymrst.params.wells_and_bc import add_well
from pymrst.utils.units import BARSA, CENTI, DARCY, DAY, MILLI, POISE


def _case(nx=4, ny=4):
    grid = compute_geometry(cart_grid([nx, ny, 1], [200.0, 200.0, 10.0]))
    rock = make_rock(grid, 100*MILLI*DARCY, 0.2)
    fluid = init_simple_adi_fluid(
        mu=[1*CENTI*POISE, 5*CENTI*POISE], rho=[1000.0, 800.0],
        n=[2.0, 2.0], c=[1e-10, 1e-9], p_ref=200*BARSA)
    wells = add_well(None, grid, rock, np.array([0]), well_type="rate",
                     val=40.0/DAY, compi=np.array([1.0, 0.0]), name="I")
    wells = add_well(wells, grid, rock, np.array([grid.cells.num - 1]),
                     well_type="bhp", val=200*BARSA,
                     compi=np.array([1.0, 0.0]), name="P")
    model = TwoPhaseOilWaterModelAD(grid, rock, fluid, wells)
    state0 = model.initial_state(200*BARSA, 0.2)
    dts = [20.0*DAY]*2
    controls = np.array([[40.0/DAY, 200*BARSA]]*len(dts))
    return model, state0, dts, controls, npv_objective()


def _objective_of(model, objective, state0, dts):
    def evaluate(controls):
        states = simulate_forward(model, state0, dts, controls)
        return evaluate_objective(model, objective, states, dts, controls)
    return evaluate


def test_the_adjoint_gradient_matches_a_central_finite_difference():
    """The property that makes the module correct or not.

    Every control of every step, so a sweep that gets the last step right and
    drops the coupling to earlier ones cannot pass.
    """
    model, state0, dts, controls, objective = _case()
    result = compute_gradient_adjoint(model, state0, dts, controls, objective)
    evaluate = _objective_of(model, objective, state0, dts)

    finite = np.zeros_like(controls)
    for step in range(len(dts)):
        for well in range(controls.shape[1]):
            h = 1e-6*max(abs(controls[step, well]), 1.0)
            up, down = controls.copy(), controls.copy()
            up[step, well] += h
            down[step, well] -= h
            finite[step, well] = (evaluate(up) - evaluate(down))/(2*h)

    relative = np.abs(result.gradient - finite)/np.maximum(np.abs(finite), 1e-30)
    assert relative.max() < 1e-5, (
        f"adjoint\n{result.gradient}\nfinite difference\n{finite}")


def test_every_step_carries_a_gradient():
    """A sweep that forgot the backward coupling still returns a full array.

    The earlier step's gradient is the one that depends on the coupling term,
    so it being zero - or equal to the last step's - is the visible symptom.
    """
    model, state0, dts, controls, objective = _case()
    gradient = compute_gradient_adjoint(
        model, state0, dts, controls, objective).gradient
    rates = gradient[:, 0]
    assert np.all(np.abs(rates) > 0.0)
    assert not np.isclose(rates[0], rates[1], rtol=1e-3)


def test_the_reported_objective_is_the_simulated_one():
    model, state0, dts, controls, objective = _case()
    result = compute_gradient_adjoint(model, state0, dts, controls, objective)
    direct = _objective_of(model, objective, state0, dts)(controls)
    assert result.value == pytest.approx(direct, rel=1e-12)


def test_a_control_row_per_timestep_is_required():
    model, state0, dts, controls, objective = _case()
    with pytest.raises(AssertionError):
        compute_gradient_adjoint(model, state0, dts, controls[:1], objective)
    with pytest.raises(AssertionError):
        compute_gradient_adjoint(model, state0, dts,
                                 np.hstack([controls, controls]), objective)


def test_supplied_states_are_used_rather_than_resimulated():
    """`states` is an optimisation, and a driver that ignores it silently
    doubles the cost of every gradient in an outer loop."""
    model, state0, dts, controls, objective = _case()
    states = simulate_forward(model, state0, dts, controls)
    reused = compute_gradient_adjoint(model, state0, dts, controls, objective,
                                      states=states)
    fresh = compute_gradient_adjoint(model, state0, dts, controls, objective)
    np.testing.assert_allclose(reused.gradient, fresh.gradient, rtol=1e-12)
    assert reused.states[-1] is states[-1]


# --------------------------------------------------------------------------
# A duplication, pinned rather than repaired

def test_the_two_public_paths_are_separate_copies_of_one_function():
    """Recorded as a defect, per F0.4 step 4, and deliberately not fixed here.

    `optimization/utils/compute_gradient_adjoint_ad.py` re-implements the
    backward sweep that already exists in `optimization/adjoint.py`; the bodies
    differ only in a docstring, one comment and line wrapping. The MRST-homing
    shim under `ad_core/simulators/` does the right thing and re-exports, so
    `pymrst.modules.optimization.compute_gradient_adjoint` and the function at
    the MRST path are *different objects* today.

    Deduplicating is a change to the replica layer, not an addition to it, so
    it needs an owner decision. What this test does instead is make the drift
    loud: the two copies agree numerically right now, and the day someone fixes
    one without the other, this fails.
    """
    from pymrst.modules.ad_core.simulators.compute_gradient_adjoint_ad import (
        compute_gradient_adjoint as at_mrst_path,
    )
    from pymrst.modules.optimization import (
        compute_gradient_adjoint as via_optimization,
    )
    assert via_optimization is not at_mrst_path, (
        "the duplication this test pins appears to have been resolved - "
        "delete this test and the ledger entry rather than weakening it"
    )

    model, state0, dts, controls, objective = _case()
    mine = via_optimization(model, state0, dts, controls, objective)
    theirs = at_mrst_path(model, state0, dts, controls, objective)
    np.testing.assert_allclose(mine.gradient, theirs.gradient, rtol=1e-12)
    assert mine.value == pytest.approx(theirs.value, rel=1e-12)


def test_both_copies_share_one_result_type():
    """The diagnostics trio's worse failure, checked for here and absent.

    There, three copies each defined their own dataclass, so identical objects
    failed isinstance across import paths. These two import `AdjointResult`
    from one place.
    """
    from pymrst.modules.ad_core.simulators.compute_gradient_adjoint_ad import (
        compute_gradient_adjoint as at_mrst_path,
    )
    from pymrst.modules.optimization import (
        compute_gradient_adjoint as via_optimization,
    )
    model, state0, dts, controls, objective = _case()
    mine = via_optimization(model, state0, dts, controls, objective)
    theirs = at_mrst_path(model, state0, dts, controls, objective)
    assert type(mine) is type(theirs)
