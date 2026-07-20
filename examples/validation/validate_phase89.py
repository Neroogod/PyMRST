"""Phase 89: calibration stress-observation derivative bridge."""

from types import SimpleNamespace

import numpy as np

from pymrst.gridprocessing import cart_grid, compute_geometry
from pymrst.modules.vemmech.ad_version import vem_lin_elast_ad
from pymrst.modules.vemmech.calibration import (
    _objective_match_value_from_stress,
    calculate_stress,
    identify_boundary_elements,
    objective_stress_gradient,
    set_bcfun,
    set_material_fun,
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


def finite_difference_gradient(stress, data, model_sigma, scaling):
    flat = np.asarray(stress, dtype=float).reshape(-1).copy()
    grad = np.zeros_like(flat)
    for i in range(flat.size):
        step = 1e-6 * max(abs(flat[i]), 1.0)
        sp = flat.copy()
        sm = flat.copy()
        sp[i] += step
        sm[i] -= step
        grad[i] = (
            _objective_match_value_from_stress(
                sp.reshape(-1, 6), data, model_sigma, scaling=scaling
            )
            - _objective_match_value_from_stress(
                sm.reshape(-1, 6), data, model_sigma, scaling=scaling
            )
        ) / (2.0 * step)
    return grad


def main() -> int:
    g = create_augmented_grid(
        compute_geometry(cart_grid([1, 1, 1], [1.0, 1.0, 1.0]))
    )
    binfo = identify_boundary_elements(g)
    bcfun, num_bc = set_bcfun(
        g,
        binfo,
        np.array([[0.18, 0.035], [0.025, 0.00], [-0.015, 0.045]]),
    )
    props = SimpleNamespace(
        young=np.full(g.cells.num, 1.3e9),
        poisson=np.full(g.cells.num, 0.22),
        rho=np.full(g.cells.num, 900.0),
    )
    mfuns, _ = set_material_fun(g, props, None)
    u0 = np.linspace(0.25, 0.85, num_bc)
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
        orientation=np.array([[1, theta[0] + 0.020, 0.10]]),
        Smax=np.array([[1, 1.018 * smax[0], 0.07]]),
        Smin=np.array([[1, 0.982 * smin[0], 0.08]]),
        Sv=np.array([[1, 1.011 * stress[0, 2], 0.09]]),
    )
    model_sigma = np.array([0.02, 1.0e5, 1.0e5, 1.0e5])
    scaling = 1.4
    analytic = objective_stress_gradient(
        stress, data, model_sigma, scaling=scaling
    )
    fd = finite_difference_gradient(stress, data, model_sigma, scaling)

    failures = 0
    print("Phase 89: calibration stress-observation derivative bridge")
    failures += compare("stress observation gradient", analytic, fd,
                        rtol=5e-5, atol=5e-8)
    failures += compare("finite analytic", [np.isfinite(analytic).all()], [1],
                        rtol=0, atol=0)
    if failures:
        print(f"\nStress-observation derivative validation failed ({failures} checks).")
    else:
        print("\nPyMRST calibration stress-observation derivatives are analytic.")
    return failures


if __name__ == "__main__":
    raise SystemExit(main())
