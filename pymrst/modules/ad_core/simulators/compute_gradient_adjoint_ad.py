"""Adjoint gradient driver, homed under ad_core/simulators (MRST
`computeGradientAdjointAD`).

MRST files the adjoint gradient driver in `ad-core/simulators/`; this port grew
it inside the `optimization` module. This module gives it its MRST home: the
single backward sweep that returns `dJ/du` for every control of every timestep
at the cost of one forward simulation. The implementation lives in
`optimization.adjoint`; here it is re-exported at the path MRST uses, alongside
`compute_gradient_perturbation` (the finite-difference cross-check) so the two
gradient routes sit side by side, as they do in MRST's simulators directory.
"""

from pymrst.modules.optimization.adjoint import (
    AdjointResult,
    Objective,
    compute_gradient_adjoint,
    evaluate_objective,
    npv_objective,
    simulate_forward,
)

__all__ = [
    "AdjointResult",
    "Objective",
    "compute_gradient_adjoint",
    "evaluate_objective",
    "npv_objective",
    "simulate_forward",
]


if __name__ == "__main__":
    import numpy as np

    from pymrst.gridprocessing import cart_grid, compute_geometry
    from pymrst.modules.ad_blackoil.models.two_phase_oil_water_model import (
        TwoPhaseOilWaterModelAD,
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

    def test_adjoint_gradient_matches_finite_difference_through_home():
        """The Phase 18 gradient check, run through the ad_core/simulators
        entry point: the adjoint gradient equals a central finite difference."""
        model, st0 = _model()
        dts = [25.0 * DAY] * 3
        u0 = np.array([[60.0 / DAY, 200 * BARSA]] * len(dts))
        obj = npv_objective()

        res = compute_gradient_adjoint(model, st0, dts, u0, obj)

        fd = np.zeros_like(u0)
        for n in range(len(dts)):
            for k in range(len(model.wells)):
                h = 1e-6 * max(abs(u0[n, k]), 1.0)
                up, dn = u0.copy(), u0.copy()
                up[n, k] += h
                dn[n, k] -= h
                ju = evaluate_objective(model, obj,
                                        simulate_forward(model, st0, dts, up),
                                        dts, up)
                jd = evaluate_objective(model, obj,
                                        simulate_forward(model, st0, dts, dn),
                                        dts, dn)
                fd[n, k] = (ju - jd) / (2 * h)

        err = np.abs(res.gradient - fd).max() / np.abs(fd).max()
        assert err < 1e-5, (res.gradient, fd)
        print(f"ok test_adjoint_gradient_matches_finite_difference_through_home "
              f"(NPV = {res.value:.3e}, max rel err = {err:.2e})")

    test_adjoint_gradient_matches_finite_difference_through_home()
    print("\nAll tests passed!")
