"""Phase 86: calibration load/density derivative columns."""

from types import SimpleNamespace

import numpy as np

from pymrst.gridprocessing import cart_grid, compute_geometry
from pymrst.modules.vemmech.ad_version import vem_lin_elast_ad, vem_reduced_system_residual
from pymrst.modules.vemmech.calibration import (
    calculate_stress,
    identify_boundary_elements,
    set_bcfun,
    set_material_fun,
    setup_objective_fun,
    sigma_to_smax_smin_theta,
)
from pymrst.modules.vemmech.vem import create_augmented_grid


def compare(name, py, ref, rtol=5e-4, atol=1e-8) -> int:
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
        np.array([[0.10, 0.03], [0.02, 0.00], [-0.02, 0.04]]),
    )
    props = SimpleNamespace(
        young=np.full(g.cells.num, 1.1e9),
        poisson=np.full(g.cells.num, 0.24),
        rho=np.full(g.cells.num, 1000.0),
    )
    matcond = np.array([[1, 0.00, 0.00, 0.25]])
    mfuns, num_m = set_material_fun(g, props, matcond, num_bc)
    u0 = np.concatenate([np.linspace(0.25, 0.75, num_bc), np.array([0.4])])
    bc = bcfun(u0)
    uu, extra = vem_lin_elast_ad(
        g, mfuns.efun(u0), mfuns.nufun(u0), bc, mfuns.loadfun(u0)
    )
    full = uu.reshape(-1)
    free = ~extra.disc["isdirdofs"]

    ljac = mfuns.load_jac(u0)
    load_col = num_bc
    h = 1e-6
    up = u0.copy()
    um = u0.copy()
    up[load_col] += h
    um[load_col] -= h
    fd_res = (
        vem_reduced_system_residual(
            g, mfuns.efun(up), mfuns.nufun(up), bcfun(up), mfuns.loadfun(up),
            full, extra=extra,
        )[0]
        - vem_reduced_system_residual(
            g, mfuns.efun(um), mfuns.nufun(um), bcfun(um), mfuns.loadfun(um),
            full, extra=extra,
        )[0]
    ) / (2.0 * h)
    analytic_res = -ljac[free, load_col].toarray().ravel()

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
    val, grad, _, _ = setup.objfun(u0)
    direction = np.linspace(-0.4, 0.8, u0.size)
    direction /= np.linalg.norm(direction)
    h = 1e-5
    fd_dir = (setup.objfun(u0 + h * direction)[0]
              - setup.objfun(u0 - h * direction)[0]) / (2.0 * h)
    adj_dir = float(grad @ direction)

    failures = 0
    print("Phase 86: calibration load/density derivative columns")
    failures += compare("num material controls", num_m, [0, 0, 1],
                        rtol=0, atol=0)
    failures += compare("reduced residual rho column", analytic_res, fd_res,
                        rtol=1e-6, atol=1e-8)
    failures += compare("objective finite", [np.isfinite(val)], [1],
                        rtol=0, atol=0)
    failures += compare("gradient directional check", [adj_dir], [fd_dir])
    if failures:
        print(f"\nLoad/density derivative validation failed ({failures} checks).")
    else:
        print("\nPyMRST calibration adjoint uses load/density derivative columns.")
    return failures


if __name__ == "__main__":
    raise SystemExit(main())
