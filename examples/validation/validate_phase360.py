"""Phase 360: adjoint gradient drivers homed under ad_core/simulators (AC18).

AC18 gives MRST's adjoint drivers their MRST home. In MRST they live in
`ad-core/simulators/` (`computeGradientAdjointAD`, `computeGradientPerturbationAD`);
this port had grown them inside the `optimization` module. Phase 360 re-exports
the adjoint at the MRST path and adds the perturbation gradient beside it.

There is no MATLAB reference here, by the nature of the check: the adjoint's
correctness is a *mathematical* property - it is exact or it is not - and the
independent ground truth is a finite-difference gradient, not another code's
number. So this validates exactly what AC18 specifies:

- the Phase 18 gradient check (adjoint vs a central finite difference) re-run
  through the `ad_core/simulators` entry point, and
- a perturbation-vs-adjoint cross-check: `compute_gradient_perturbation`
  (`computeGradientPerturbationAD`) confirms `compute_gradient_adjoint`
  control for control, on an NPV objective over a three-step waterflood.

Both routes share one forward simulation's objective value, so that is checked
too.
"""

import numpy as np

from pymrst.gridprocessing import cart_grid, compute_geometry
from pymrst.modules.ad_blackoil.models.two_phase_oil_water_model import (
    TwoPhaseOilWaterModelAD,
)
from pymrst.modules.ad_core.simulators.compute_gradient_adjoint_ad import (
    compute_gradient_adjoint,
    evaluate_objective,
    npv_objective,
    simulate_forward,
)
from pymrst.modules.ad_core.simulators.compute_gradient_perturbation_ad import (
    compute_gradient_perturbation,
)
from pymrst.modules.ad_props.init_simple_adi_fluid import init_simple_adi_fluid
from pymrst.params.rock import make_rock
from pymrst.params.wells_and_bc import add_well
from pymrst.utils.units import BARSA, CENTI, DARCY, DAY, MILLI, POISE


def _check(name: str, ok: bool, detail: str = "") -> int:
    print(f"  {'ok  ' if ok else 'FAIL'} {name}{('  ' + detail) if detail else ''}")
    return 0 if ok else 1


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
    return TwoPhaseOilWaterModelAD(g, rock, fluid, W), None


def main() -> int:
    model, _ = _model()
    st0 = model.initial_state(200 * BARSA, 0.2)
    dts = [25.0 * DAY] * 3
    u0 = np.array([[60.0 / DAY, 200 * BARSA]] * len(dts))
    obj = npv_objective()

    failed = 0

    adj = compute_gradient_adjoint(model, st0, dts, u0, obj)

    # -- adjoint vs a hand-rolled central finite difference (the Phase 18 check)
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
    err_fd = np.abs(adj.gradient - fd).max() / np.abs(fd).max()

    print("Adjoint vs finite difference (through ad_core/simulators):")
    failed += _check("adjoint gradient matches central FD", err_fd < 1e-5,
                     f"max rel err = {err_fd:.2e}")

    # -- perturbation driver vs adjoint driver (the cross-check AC18 asks for) --
    pert = compute_gradient_perturbation(model, st0, dts, u0, obj)
    err_pert = np.abs(adj.gradient - pert.gradient).max() / np.abs(pert.gradient).max()

    print("\nPerturbation gradient vs adjoint gradient:")
    failed += _check("objective value agrees",
                     np.isclose(adj.value, pert.value, rtol=1e-10),
                     f"NPV = {adj.value:.6e}")
    failed += _check("gradient matches control for control", err_pert < 1e-5,
                     f"max rel err = {err_pert:.2e}")

    # forward-difference (MRST's literal form) agrees a touch looser
    pertf = compute_gradient_perturbation(model, st0, dts, u0, obj, central=False)
    err_fwd = np.abs(adj.gradient - pertf.gradient).max() / np.abs(pertf.gradient).max()
    failed += _check("forward-difference form (central=False) agrees too",
                     err_fwd < 1e-3, f"max rel err = {err_fwd:.2e}")

    print("\nAdjoint drivers agree with finite differences!" if failed == 0
          else f"\n{failed} checks FAILED")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
