"""Phase 90: calibration eliminated-objective ADI2 Hessian output."""

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


def compare(name, py, ref, rtol=1e-5, atol=1e-8) -> int:
    py = np.asarray(py, dtype=float)
    ref = np.asarray(ref, dtype=float)
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
        np.array([[0.09, 0.025], [0.015, 0.00], [-0.018, 0.035]]),
    )
    props = SimpleNamespace(
        young=np.full(g.cells.num, 1.05e9),
        poisson=np.full(g.cells.num, 0.24),
        rho=np.full(g.cells.num, 990.0),
    )
    mfuns, _ = set_material_fun(g, props, None)
    u0 = np.linspace(0.30, 0.70, num_bc)
    uu, extra = vem_lin_elast_ad(
        g, mfuns.efun(u0), mfuns.nufun(u0), bcfun(u0), mfuns.loadfun(u0)
    )
    stress = calculate_stress(
        u0, uu.reshape(-1), g, bcfun, mfuns.efun, mfuns.nufun, extra
    ).reshape(-1, 6)
    smax, smin, theta = sigma_to_smax_smin_theta(
        stress[:, 0], stress[:, 1], stress[:, 3]
    )
    data = SimpleNamespace(
        orientation=np.array([[1, theta[0] + 0.011, 0.10]]),
        Smax=np.array([[1, 1.009 * smax[0], 0.10]]),
        Smin=np.array([[1, 0.991 * smin[0], 0.10]]),
        Sv=np.array([[1, stress[0, 2], 0.10]]),
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
    val, grad, qhess, qhess_dir = setup.objfun_with_hessian(u0)
    old_val, old_grad, old_qhess, old_qhess_dir = setup.objfun(u0)

    fd = np.zeros_like(qhess)
    for i in range(u0.size):
        step = 1e-5 * max(abs(u0[i]), 1.0)
        up = u0.copy()
        um = u0.copy()
        up[i] += step
        um[i] -= step
        gp = setup.objfun(up)[1]
        gm = setup.objfun(um)[1]
        fd[:, i] = (gp - gm) / (2.0 * step)
    fd = 0.5 * (fd + fd.T)
    direction = np.linspace(-0.3, 0.7, u0.size)
    direction /= np.linalg.norm(direction)

    failures = 0
    print("Phase 90: calibration eliminated-objective ADI2 Hessian")
    failures += compare("value compatibility", [val], [old_val], rtol=1e-12)
    failures += compare("gradient compatibility", grad, old_grad, rtol=1e-12)
    failures += compare("objfun keeps empty Hessian", [old_qhess is None], [1],
                        rtol=0, atol=0)
    failures += compare("objfun keeps empty Hessian direction",
                        [old_qhess_dir is None], [1], rtol=0, atol=0)
    failures += compare("ADI2 Hessian vs FD reference", qhess, fd,
                        rtol=1e-10, atol=1e-10)
    failures += compare("ADI2 Hessian symmetry", qhess, qhess.T,
                        rtol=1e-12, atol=1e-12)
    failures += compare("ADI2 Hessian direction", qhess_dir(direction),
                        qhess @ direction, rtol=1e-12, atol=1e-12)
    if failures:
        print(f"\nADI2 Hessian validation failed ({failures} checks).")
    else:
        print("\nPyMRST calibration exposes optional ADI2 Hessian output.")
    return failures


if __name__ == "__main__":
    raise SystemExit(main())
