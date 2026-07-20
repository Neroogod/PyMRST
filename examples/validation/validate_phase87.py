"""Phase 87: calibration objective displacement partials."""

from types import SimpleNamespace

import numpy as np

from pymrst.gridprocessing import cart_grid, compute_geometry
from pymrst.modules.vemmech.ad_version import vem_lin_elast_ad
from pymrst.modules.vemmech.calibration import (
    calculate_stress,
    identify_boundary_elements,
    objective_displacement_gradient,
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
        np.array([[0.12, 0.025], [0.015, 0.00], [-0.018, 0.035]]),
    )
    props = SimpleNamespace(
        young=np.full(g.cells.num, 1.15e9),
        poisson=np.full(g.cells.num, 0.23),
        rho=np.full(g.cells.num, 950.0),
    )
    mfuns, num_m = set_material_fun(g, props, None)
    u0 = np.linspace(0.20, 0.80, num_bc)
    uu, extra = vem_lin_elast_ad(
        g, mfuns.efun(u0), mfuns.nufun(u0), bcfun(u0), mfuns.loadfun(u0)
    )
    full = uu.reshape(-1)
    free = ~extra.disc["isdirdofs"]

    stress = calculate_stress(
        u0, full, g, bcfun, mfuns.efun, mfuns.nufun, extra
    ).reshape(-1, 6)
    smax, smin, theta = sigma_to_smax_smin_theta(
        stress[:, 0], stress[:, 1], stress[:, 3]
    )
    data = SimpleNamespace(
        orientation=np.array([[1, theta[0] + 0.010, 0.12]]),
        Smax=np.array([[1, 1.012 * smax[0], 0.08]]),
        Smin=np.array([[1, 0.988 * smin[0], 0.08]]),
        Sv=np.array([[1, 1.006 * stress[0, 2], 0.08]]),
    )
    setup = setup_objective_fun(
        g,
        data,
        bcfun,
        mfuns,
        num_bc,
        np.array([]),
        u0,
        np.ones(num_bc),
        np.zeros(4),
        scaling=1.0,
        ignore_prior=False,
    )
    grad_full = objective_displacement_gradient(
        u0, full, extra, g, data, bcfun, mfuns.efun, mfuns.nufun,
        np.zeros(4), None, 1.0,
    )
    grad_free = grad_full[free]

    fd = np.zeros(np.count_nonzero(free))
    for local, dof in enumerate(np.flatnonzero(free)):
        step = 1e-6 * max(abs(full[dof]), 1.0)
        up = full.copy()
        um = full.copy()
        up[dof] += step
        um[dof] -= step
        fd[local] = (setup.ofun(u0, up, extra)[0]
                     - setup.ofun(u0, um, extra)[0]) / (2.0 * step)

    val, grad, _, _ = setup.objfun(u0)
    direction = np.linspace(-0.2, 0.9, u0.size)
    direction /= np.linalg.norm(direction)
    h = 1e-5
    fd_dir = (setup.objfun(u0 + h * direction)[0]
              - setup.objfun(u0 - h * direction)[0]) / (2.0 * h)
    adj_dir = float(grad @ direction)

    failures = 0
    print("Phase 87: calibration objective displacement partials")
    failures += compare("num material controls", num_m, [0, 0, 0], rtol=0, atol=0)
    failures += compare("objective displacement partial", grad_free, fd,
                        rtol=2e-5, atol=2e-7)
    failures += compare("objective finite", [np.isfinite(val)], [1],
                        rtol=0, atol=0)
    failures += compare("gradient directional check", [adj_dir], [fd_dir],
                        rtol=5e-4, atol=1e-6)
    if failures:
        print(f"\nObjective displacement partial validation failed ({failures} checks).")
    else:
        print("\nPyMRST calibration objective uses stress-operator displacement partials.")
    return failures


if __name__ == "__main__":
    raise SystemExit(main())
