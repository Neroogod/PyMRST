"""Finite-difference gradient of a schedule objective (MRST
`computeGradientPerturbationAD`).

The brute-force counterpart to the adjoint: perturb one control at a time,
re-simulate the whole schedule, and difference the objective. It costs one
forward solve per control - `O(n_steps * n_wells)` simulations against the
adjoint's single backward sweep - so it is never used for large problems. Its
job is to be the *ground truth*: MRST ships it next to `computeGradientAdjointAD`
precisely so a perturbation gradient can confirm an adjoint gradient, and that is
what it does here.

MRST perturbs each control by an absolute `perturbation` scaled by control type,
which needs the caller to supply sensible scaling because a rate and a bhp differ
by orders of magnitude. This port uses a *relative* step,
`perturbation * max(|u|, 1)`, so the routine is accurate across both scales
without hand-tuned scaling, and defaults to a central difference for a tighter
check. The forward-difference, absolute-step MRST form is available via
`central=False` and an explicit `step`.
"""

from typing import Callable, Optional, Sequence

import numpy as np

from pymrst.modules.optimization.adjoint import (
    AdjointResult,
    Objective,
    evaluate_objective,
    simulate_forward,
)


def compute_gradient_perturbation(
    model,
    state0,
    dts: Sequence[float],
    controls: np.ndarray,
    objective: Objective,
    perturbation: float = 1e-6,
    central: bool = True,
    step: Optional[Callable[[int, int, float], float]] = None,
) -> AdjointResult:
    """`dJ/du` by finite differences over the controls.

    Perturbs each control `(n, k)` independently and re-simulates. `step(n, k,
    u)` gives the perturbation for that control; the default is the relative
    `perturbation * max(|u|, 1)`. Returns the same `AdjointResult` shape as the
    adjoint so the two can be compared directly.
    """
    controls = np.atleast_2d(np.asarray(controls, float))
    nsteps, nw = controls.shape
    assert len(dts) == nsteps, "one control row per timestep"
    assert nw == len(model.wells), "one control per well"

    if step is None:
        def step(n, k, u):
            return perturbation * max(abs(u), 1.0)

    states0 = simulate_forward(model, state0, dts, controls)
    val0 = evaluate_objective(model, objective, states0, dts, controls)

    grad = np.zeros((nsteps, nw))
    for n in range(nsteps):
        for k in range(nw):
            e = step(n, k, controls[n, k])
            up = controls.copy()
            up[n, k] += e
            ju = evaluate_objective(model, objective,
                                    simulate_forward(model, state0, dts, up),
                                    dts, up)
            if central:
                dn = controls.copy()
                dn[n, k] -= e
                jd = evaluate_objective(model, objective,
                                        simulate_forward(model, state0, dts, dn),
                                        dts, dn)
                grad[n, k] = (ju - jd) / (2 * e)
            else:
                grad[n, k] = (ju - val0) / e

    return AdjointResult(value=val0, gradient=grad, states=states0)


#: MRST's own spelling. This port names functions in snake_case; the
#: alias keeps the MATLAB name callable, which is the convention the
#: rest of the package already follows.
computeGradientPerturbationAD = compute_gradient_perturbation

if __name__ == "__main__":
    from pymrst.gridprocessing import cart_grid, compute_geometry
    from pymrst.modules.ad_blackoil.models.two_phase_oil_water_model import (
        TwoPhaseOilWaterModelAD,
    )
    from pymrst.modules.ad_core.simulators.compute_gradient_adjoint_ad import (
        compute_gradient_adjoint,
        npv_objective,
    )
    from pymrst.modules.ad_props.init_simple_adi_fluid import init_simple_adi_fluid
    from pymrst.params.rock import make_rock
    from pymrst.params.wells_and_bc import add_well
    from pymrst.utils.units import BARSA, CENTI, DARCY, DAY, MILLI, POISE

    def _model(nx=5, ny=5):
        g = compute_geometry(cart_grid([nx, ny, 1], [300.0, 300.0, 10.0]))
        rock = make_rock(g, 100 * MILLI * DARCY, 0.2)
        fluid = init_simple_adi_fluid(
            mu=[1 * CENTI * POISE, 5 * CENTI * POISE], rho=[1000.0, 800.0],
            n=[2.0, 2.0], c=[1e-10, 1e-9], p_ref=200 * BARSA)
        W = add_well(None, g, rock, np.array([0]), well_type="rate",
                     val=60.0 / DAY, compi=np.array([1.0, 0.0]), name="I")
        W = add_well(W, g, rock, np.array([g.cells.num - 1]), well_type="bhp",
                     val=200 * BARSA, compi=np.array([1.0, 0.0]), name="P")
        model = TwoPhaseOilWaterModelAD(g, rock, fluid, W)
        return model, model.initial_state(200 * BARSA, 0.2)

    def test_perturbation_matches_adjoint():
        """The whole point: the finite-difference gradient confirms the adjoint
        gradient, control for control."""
        import numpy as np
        model, st0 = _model()
        dts = [25.0 * DAY] * 3
        u0 = np.array([[60.0 / DAY, 200 * BARSA]] * len(dts))
        obj = npv_objective()

        adj = compute_gradient_adjoint(model, st0, dts, u0, obj)
        pert = compute_gradient_perturbation(model, st0, dts, u0, obj)

        assert np.isclose(adj.value, pert.value, rtol=1e-10)
        err = np.abs(adj.gradient - pert.gradient).max() / np.abs(pert.gradient).max()
        assert err < 1e-5, (adj.gradient, pert.gradient)
        print(f"ok test_perturbation_matches_adjoint (max rel err = {err:.2e})")

    def test_forward_difference_form_also_agrees():
        """MRST's forward-difference form (central=False) agrees too, a touch
        looser as forward differences are only first-order."""
        import numpy as np
        model, st0 = _model(4, 4)
        dts = [20.0 * DAY] * 2
        u0 = np.array([[50.0 / DAY, 200 * BARSA]] * len(dts))
        obj = npv_objective()

        adj = compute_gradient_adjoint(model, st0, dts, u0, obj)
        pert = compute_gradient_perturbation(model, st0, dts, u0, obj,
                                             central=False)
        err = np.abs(adj.gradient - pert.gradient).max() / np.abs(pert.gradient).max()
        assert err < 1e-3, (adj.gradient, pert.gradient)
        print(f"ok test_forward_difference_form_also_agrees (max rel err = {err:.2e})")

    test_perturbation_matches_adjoint()
    test_forward_difference_form_also_agrees()
    print("\nAll tests passed!")
