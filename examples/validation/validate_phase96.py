"""Phase 96: calibration Hessians with material controls."""

from types import SimpleNamespace

import numpy as np

from pymrst.gridprocessing import cart_grid, compute_geometry
from pymrst.modules.vemmech.ad_version import vem_lin_elast_ad
from pymrst.modules.vemmech.calibration import (
    calculate_stress,
    identify_boundary_elements,
    set_bcfun,
    set_material_fun,
    setup_objective_fun,
    sigma_to_smax_smin_theta,
)
from pymrst.modules.vemmech.vem import create_augmented_grid


def compare(name, py, ref, rtol=1e-8, atol=1e-8) -> int:
    py = np.asarray(py, dtype=float)
    ref = np.asarray(ref, dtype=float)
    scale = max(float(np.max(np.abs(ref))) if ref.size else 0.0, 1.0)
    diff = 0.0 if py.size == ref.size == 0 else float(np.max(np.abs(py - ref)))
    ok = py.shape == ref.shape and diff <= atol + rtol * scale
    print(f"  {'ok  ' if ok else 'FAIL'} {name}: max abs diff = {diff:.3e} "
          f"(rel {diff / scale:.2e}, rtol {rtol:.0e})")
    return 0 if ok else 1


def _setup_case(matcond, mat_controls):
    g = create_augmented_grid(
        compute_geometry(cart_grid([1, 1, 1], [1.0, 1.0, 1.0]))
    )
    binfo = identify_boundary_elements(g)
    bcfun, num_bc = set_bcfun(
        g,
        binfo,
        np.array([[0.10, 0.03], [0.02, 0.00], [-0.02, 0.04]]),
    )
    props = SimpleNamespace(
        young=np.full(g.cells.num, 1.1e9),
        poisson=np.full(g.cells.num, 0.24),
        rho=np.full(g.cells.num, 1000.0),
    )
    mfuns, num_m = set_material_fun(g, props, matcond, num_bc)
    u0 = np.concatenate([np.linspace(0.25, 0.75, num_bc), mat_controls])
    uu, extra = vem_lin_elast_ad(
        g, mfuns.efun(u0), mfuns.nufun(u0), bcfun(u0), mfuns.loadfun(u0)
    )
    stress = calculate_stress(
        u0, uu, g, bcfun, mfuns.efun, mfuns.nufun, extra
    ).reshape(-1, 6)
    smax, smin, theta = sigma_to_smax_smin_theta(
        stress[:, 0], stress[:, 1], stress[:, 3]
    )
    data = SimpleNamespace(
        orientation=np.array([[1, theta[0] + 0.012, 0.10]]),
        Smax=np.array([[1, 1.015 * smax[0], 0.10]]),
        Smin=np.array([[1, 0.985 * smin[0], 0.10]]),
        Sv=np.array([[1, stress[0, 2], 0.10]]),
    )
    setup = setup_objective_fun(
        g,
        data,
        bcfun,
        mfuns,
        num_bc,
        matcond[:, 1:].reshape(-1),
        u0,
        np.ones(u0.size),
        np.zeros(4),
        scaling=1.0,
        ignore_prior=False,
    )
    return setup, u0, num_m


def _fd_gradient_hessian(setup, u0):
    fd = np.zeros((u0.size, u0.size))
    for i in range(u0.size):
        step = 1e-5 * max(abs(u0[i]), 1.0)
        up = u0.copy()
        um = u0.copy()
        up[i] += step
        um[i] -= step
        gp = setup.objfun(up)[1]
        gm = setup.objfun(um)[1]
        fd[:, i] = (gp - gm) / (2.0 * step)
    return 0.5 * (fd + fd.T)


def _check_case(name, matcond, mat_controls, expected_num_m):
    setup, u0, num_m = _setup_case(matcond, mat_controls)
    val, grad, hess, hess_dir = setup.objfun_with_hessian(u0)
    fd = _fd_gradient_hessian(setup, u0)
    direction = np.linspace(-0.3, 0.7, u0.size)
    direction /= np.linalg.norm(direction)

    failures = 0
    print(f"Phase 96: {name}")
    failures += compare("num material controls", num_m, expected_num_m,
                        rtol=0, atol=0)
    failures += compare("value finite", [np.isfinite(val)], [1], rtol=0, atol=0)
    failures += compare("gradient finite", [np.isfinite(grad).all()], [1],
                        rtol=0, atol=0)
    failures += compare("material Hessian vs FD reference", hess, fd,
                        rtol=1e-10, atol=1e-10)
    failures += compare("material Hessian symmetry", hess, hess.T,
                        rtol=1e-12, atol=1e-12)
    failures += compare("material Hessian direction", hess_dir(direction),
                        hess @ direction, rtol=1e-12, atol=1e-12)
    return failures


def main() -> int:
    failures = 0
    failures += _check_case(
        "K/G stiffness-control Hessian",
        np.array([[1, 0.20, 0.30, 0.00]]),
        np.array([0.35, 0.65]),
        [1, 1, 0],
    )
    failures += _check_case(
        "rho load-control Hessian",
        np.array([[1, 0.00, 0.00, 0.25]]),
        np.array([0.4]),
        [0, 0, 1],
    )
    if failures:
        print(f"\nMaterial-control Hessian validation failed "
              f"({failures} checks).")
    else:
        print("\nPyMRST calibration Hessians include material-control paths.")
    return failures


if __name__ == "__main__":
    raise SystemExit(main())
