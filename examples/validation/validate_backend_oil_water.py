"""Validate the experimental Torch backend oil-water residual/Newton path."""

from __future__ import annotations

import numpy as np

from pymrst.backends import finite_difference_jacobian
from pymrst.gridprocessing import cart_grid, compute_geometry
from pymrst.modules.ad_blackoil import BackendOWFluid, BackendOilWaterModel
from pymrst.params.rock import make_rock
from pymrst.params.wells_and_bc import add_source
from pymrst.utils.units import BARSA, CENTI, DARCY, MILLI, POISE


def _model():
    g = compute_geometry(cart_grid([4, 1, 1], [40.0, 10.0, 10.0]))
    rock = make_rock(g, 100 * MILLI * DARCY, 0.25)
    src = add_source(None, np.array([0]), 1e-4, sat=np.array([1.0, 0.0]))
    src = add_source(src, np.array([g.cells.num - 1]), -1e-4, sat=np.array([0.0, 1.0]))
    fluid = BackendOWFluid.simple(
        mu=np.array([1.0, 5.0]) * CENTI * POISE,
        n=(2.0, 2.0),
        c=(1e-6 / BARSA, 1e-5 / BARSA),
        p_ref=200 * BARSA,
    )
    model = BackendOilWaterModel(
        g,
        rock,
        fluid,
        src=src,
        pressure_anchor=200 * BARSA,
        backend="torch",
        device="cpu",
    )
    return model


def main() -> int:
    try:
        import torch
    except ImportError:
        print("skip backend oil-water: torch is not installed")
        return 0

    model = _model()
    state0 = model.initial_state(200 * BARSA, 0.2)
    residual = model.residual_torch(state0, 10.0)
    x0 = model.pack(state0)
    xt = torch.as_tensor(x0, dtype=torch.float64).requires_grad_(True)
    jac_ad = torch.autograd.functional.jacobian(residual, xt, vectorize=True).detach().numpy()

    def residual_np(x):
        return residual(torch.as_tensor(x, dtype=torch.float64)).detach().numpy()

    jac_fd = finite_difference_jacobian(residual_np, x0, eps=1e-4)
    jdiff = float(np.max(np.abs(jac_ad - jac_fd)))
    ok_jac = jdiff < 2e-5
    print(f"{'ok  ' if ok_jac else 'FAIL'} autograd Jacobian: max abs diff {jdiff:.3e}")

    state1, report = model.step(state0, 10.0, max_iterations=12, tolerance=1e-8)
    final_norm = report.residual_norms[-1]
    sat_ok = bool(np.all(state1.sW > -1e-8) and np.all(state1.sW < 1.0 + 1e-8))
    ok_newton = report.converged and final_norm < 1e-8 and sat_ok
    print(
        f"{'ok  ' if ok_newton else 'FAIL'} backend oil-water Newton: "
        f"iterations {report.iterations}, residual {final_norm:.3e}, "
        f"sW range {state1.sW.min():.3f}-{state1.sW.max():.3f}"
    )
    return 0 if ok_jac and ok_newton else 1


if __name__ == "__main__":
    raise SystemExit(main())
