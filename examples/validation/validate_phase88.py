"""Phase 88: calibration objective control partials."""

from types import SimpleNamespace

import numpy as np

from pymrst.gridprocessing import cart_grid, compute_geometry
from pymrst.modules.vemmech.ad_version import vem_lin_elast_ad
from pymrst.modules.vemmech.calibration import (
    calculate_stress,
    identify_boundary_elements,
    objective_control_gradient,
    set_bcfun,
    set_material_fun,
    setup_objective_fun,
    sigma_to_smax_smin_theta,
)
from pymrst.modules.vemmech.vem import create_augmented_grid


def compare(name, py, ref, rtol=1e-5, atol=1e-8) -> int:
    py = np.asarray(py, dtype=float).ravel()
    ref = np.asarray(ref, dtype=float).ravel()
    scale = max(float(np.max(np.abs(ref))) if ref.size else 0.0, 1.0)
    diff = 0.0 if py.size == ref.size == 0 else float(np.max(np.abs(py - ref)))
    ok = py.shape == ref.shape and diff <= atol + rtol * scale
    print(f"  {'ok  ' if ok else 'FAIL'} {name}: max abs diff = {diff:.3e} "
          f"(rel {diff / scale:.2e}, rtol {rtol:.0e})")
    return 0 if ok else 1


def main() -> int:
    g = create_augmented_grid(
        compute_geometry(cart_grid([1, 1, 1], [1.0, 1.0, 1.0]))
    )
    binfo = identify_boundary_elements(g)
    bcfun, num_bc = set_bcfun(
        g,
        binfo,
        np.array([[0.11, 0.030], [0.018, 0.00], [-0.020, 0.040]]),
    )
    props = SimpleNamespace(
        young=np.full(g.cells.num, 1.20e9),
        poisson=np.full(g.cells.num, 0.24),
        rho=np.full(g.cells.num, 980.0),
    )
    matcond = np.array([[1, 0.18, 0.12, 0.20]])
    mfuns, num_m = set_material_fun(g, props, matcond, num_bc)
    u_prior = np.array([0.20, 0.35, 0.10, 0.15, 0.25])
    u0 = np.array([0.42, 0.68, 0.35, 0.45, 0.55])
    u_sigma = np.array([0.7, 0.8, 0.9, 1.0, 1.1])

    uu, extra = vem_lin_elast_ad(
        g, mfuns.efun(u0), mfuns.nufun(u0), bcfun(u0), mfuns.loadfun(u0)
    )
    full = uu.reshape(-1)
    stress = calculate_stress(
        u0, full, g, bcfun, mfuns.efun, mfuns.nufun, extra
    ).reshape(-1, 6)
    smax, smin, theta = sigma_to_smax_smin_theta(
        stress[:, 0], stress[:, 1], stress[:, 3]
    )
    data = SimpleNamespace(
        orientation=np.array([[1, theta[0] + 0.014, 0.11]]),
        Smax=np.array([[1, 1.014 * smax[0], 0.09]]),
        Smin=np.array([[1, 0.986 * smin[0], 0.09]]),
        Sv=np.array([[1, 1.004 * stress[0, 2], 0.09]]),
    )
    mat_facs = matcond[:, 1:].reshape(-1)
    setup = setup_objective_fun(
        g,
        data,
        bcfun,
        mfuns,
        num_bc,
        mat_facs,
        u_prior,
        u_sigma,
        np.zeros(4),
        scaling=1.25,
        ignore_prior=False,
    )
    grad = objective_control_gradient(
        u0,
        full,
        extra,
        g,
        data,
        bcfun,
        mfuns.efun,
        mfuns.nufun,
        num_bc,
        mat_facs,
        u_prior,
        u_sigma,
        np.zeros(4),
        scaling=1.25,
        ignore_prior=False,
        alpha_p=None,
        num_material_ctrls=mfuns.num_ctrls,
    )

    fd = np.zeros_like(u0)
    for i in range(u0.size):
        step = 1e-6 * max(abs(u0[i]), 1.0)
        up = u0.copy()
        um = u0.copy()
        up[i] += step
        um[i] -= step
        fd[i] = (setup.ofun(up, full, extra)[0]
                 - setup.ofun(um, full, extra)[0]) / (2.0 * step)

    val, adj_grad, _, _ = setup.objfun(u0)
    direction = np.linspace(-0.4, 0.6, u0.size)
    direction /= np.linalg.norm(direction)
    h = 1e-5
    fd_dir = (setup.objfun(u0 + h * direction)[0]
              - setup.objfun(u0 - h * direction)[0]) / (2.0 * h)
    adj_dir = float(adj_grad @ direction)

    failures = 0
    print("Phase 88: calibration objective control partials")
    failures += compare("num material controls", num_m, [1, 1, 1], rtol=0, atol=0)
    failures += compare("objective control partial", grad, fd,
                        rtol=2e-5, atol=2e-7)
    failures += compare("objective finite", [np.isfinite(val)], [1],
                        rtol=0, atol=0)
    failures += compare("gradient directional check", [adj_dir], [fd_dir],
                        rtol=5e-4, atol=1e-6)
    if failures:
        print(f"\nObjective control partial validation failed ({failures} checks).")
    else:
        print("\nPyMRST calibration objective uses structured control partials.")
    return failures


if __name__ == "__main__":
    raise SystemExit(main())
