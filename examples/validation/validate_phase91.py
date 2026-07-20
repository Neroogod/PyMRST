"""Phase 91: calibration workflow driver."""

from types import SimpleNamespace

import numpy as np

from pymrst.gridprocessing import cart_grid, compute_geometry
from pymrst.modules.vemmech.calibration import (
    calibrate_model,
    calculate_stress,
    identify_boundary_elements,
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
    props = SimpleNamespace(
        young=np.full(g.cells.num, 1.1e9),
        poisson=np.full(g.cells.num, 0.24),
        rho=np.full(g.cells.num, 1000.0),
    )
    bcond = np.array([[0.08, 0.030], [0.018, 0.00], [-0.016, 0.040]])
    matcond = np.zeros((0, 4))
    target = np.array([0.72, 0.24])

    synthetic = calibrate_model(
        g,
        props,
        bcond,
        matcond,
        SimpleNamespace(),
        produce_syndata_for=target,
    )
    stress = calculate_stress(
        target,
        synthetic.uu.reshape(-1),
        g,
        synthetic.bcfun,
        synthetic.mfuns.efun,
        synthetic.mfuns.nufun,
        synthetic.extra,
    ).reshape(-1, 6)
    smax, smin, theta = sigma_to_smax_smin_theta(
        stress[:, 0], stress[:, 1], stress[:, 3]
    )
    data = SimpleNamespace(
        orientation=np.array([[1, theta[0], 0.03]]),
        Smax=np.array([[1, smax[0], 0.05]]),
        Smin=np.array([[1, smin[0], 0.05]]),
        Sv=np.array([[1, stress[0, 2], 0.05]]),
    )

    binfo = identify_boundary_elements(g)
    bcfun, num_bc = set_bcfun(g, binfo, bcond)
    mfuns, _ = set_material_fun(g, props, matcond, num_bc)
    setup = setup_objective_fun(
        g,
        data,
        bcfun,
        mfuns,
        num_bc,
        np.array([]),
        np.zeros(num_bc),
        np.ones(num_bc),
        np.zeros(4),
        scaling=1.0,
        ignore_prior=True,
    )
    start = np.array([0.15, 0.85])
    start_val = setup.objfun(start)[0]
    target_val = setup.objfun(target)[0]
    res = calibrate_model(
        g,
        props,
        bcond,
        matcond,
        data,
        ignore_prior=True,
        u_start=start,
        max_it=20,
        grad_tol=1e-8,
        obj_change_tol=1e-12,
        line_search_max_it=10,
    )
    opt_val = res.obj_fun(res.u_opt)[0]

    failures = 0
    print("Phase 91: calibration workflow driver")
    failures += compare("synthetic displacement finite",
                        [np.isfinite(synthetic.uu).all()], [1], rtol=0, atol=0)
    failures += compare("num boundary controls", [res.num_bc_ctrls], [2],
                        rtol=0, atol=0)
    failures += compare("objective bookkeeping", [res.foptval], [opt_val],
                        rtol=1e-12, atol=1e-12)
    if not (res.foptval < start_val):
        print(f"  FAIL objective reduction: start={start_val:.3e}, "
              f"target={target_val:.3e}, opt={res.foptval:.3e}")
        failures += 1
    else:
        print(f"  ok   objective reduction: start={start_val:.3e}, "
              f"target={target_val:.3e}, opt={res.foptval:.3e}")
    failures += compare("history final value", [res.history.values[-1]],
                        [res.foptval], rtol=1e-12, atol=1e-12)
    failures += compare("optimal displacement finite",
                        [np.isfinite(res.uu_opt).all()], [1], rtol=0, atol=0)
    if failures:
        print(f"\nCalibration workflow validation failed ({failures} checks).")
    else:
        print("\nPyMRST calibration workflow runs synthetic data and optimization.")
    return failures


if __name__ == "__main__":
    raise SystemExit(main())
