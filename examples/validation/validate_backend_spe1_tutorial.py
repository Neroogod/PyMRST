"""Validate the backend SPE1 tutorial runner against the ADI tutorial."""

from __future__ import annotations

import numpy as np

from pymrst.modules.ad_blackoil.examples.spe1.blackoil_tutorial_spe1 import (
    run_backend_blackoil_tutorial_spe1,
    run_blackoil_tutorial_spe1,
)


def _maxdiff(a, b):
    return float(np.max(np.abs(np.asarray(a) - np.asarray(b))))


def main() -> int:
    try:
        import torch  # noqa: F401
    except ImportError:
        print("skip backend SPE1 tutorial: torch is not installed")
        return 0

    nsteps = 3
    adi = run_blackoil_tutorial_spe1(nsteps=nsteps, verbose=False)
    backend = run_backend_blackoil_tutorial_spe1(nsteps=nsteps, verbose=False)
    checks = [
        ("pressure", backend.history["pressure"], adi.history["pressure"], 8e-2),
        ("sW", backend.history["sW"], adi.history["sW"], 1e-10),
        ("sG", backend.history["sG"], adi.history["sG"], 1e-10),
        ("rs", backend.history["rs"], adi.history["rs"], 1e-7),
        ("qWs", backend.history["qWs"], adi.history["qWs"], 1e-11),
        ("qOs", backend.history["qOs"], adi.history["qOs"], 1e-11),
        ("qGs", backend.history["qGs"], adi.history["qGs"], 1e-11),
        ("bhp", backend.history["bhp"], adi.history["bhp"], 1e-1),
    ]
    failed = 0
    for name, got, ref, tol in checks:
        diff = _maxdiff(got, ref)
        ok = diff <= tol
        print(f"{'ok  ' if ok else 'FAIL'} {name}: max abs diff {diff:.3e}")
        failed += 0 if ok else 1
    for i, (bstate, astate) in enumerate(zip(backend.states, adi.states), start=1):
        ok = np.array_equal(bstate.status, astate["status"])
        print(f"{'ok  ' if ok else 'FAIL'} status step {i}")
        failed += 0 if ok else 1

    sparse = run_backend_blackoil_tutorial_spe1(
        nsteps=1,
        verbose=False,
        linear_solver="scipy_sparse",
    )
    dense_first = run_backend_blackoil_tutorial_spe1(nsteps=1, verbose=False)
    sparse_diff = _maxdiff(sparse.history["pressure"], dense_first.history["pressure"])
    sparse_ok = sparse_diff <= 1e-2
    print(f"{'ok  ' if sparse_ok else 'FAIL'} scipy sparse Newton solve: pressure diff {sparse_diff:.3e}")
    failed += 0 if sparse_ok else 1

    mf = run_backend_blackoil_tutorial_spe1(
        nsteps=1,
        verbose=False,
        linear_solver="matrix_free_gmres_fallback",
        gmres_tolerance=1e-6,
        gmres_max_iterations=120,
        gmres_restart=60,
    )
    mf_diff = _maxdiff(mf.history["pressure"], dense_first.history["pressure"])
    mf_tried = bool(mf.report[0].linear_iterations)
    mf_ok = mf_tried and mf_diff <= 1e-2
    print(
        f"{'ok  ' if mf_ok else 'FAIL'} matrix-free GMRES fallback: "
        f"pressure diff {mf_diff:.3e}, "
        f"linear attempts {mf.report[0].linear_iterations}"
    )
    failed += 0 if mf_ok else 1

    pure_mf = run_backend_blackoil_tutorial_spe1(
        nsteps=1,
        verbose=False,
        linear_solver="matrix_free_gmres",
        gmres_scaling="variable",
        gmres_tolerance=1e-3,
        gmres_max_iterations=240,
        gmres_restart=120,
    )
    pure_mf_diff = _maxdiff(pure_mf.history["pressure"], dense_first.history["pressure"])
    pure_mf_bhp = _maxdiff(pure_mf.history["bhp"], dense_first.history["bhp"])
    pure_mf_ok = pure_mf.report[0].converged and pure_mf_diff <= 1.0 and pure_mf_bhp <= 2.0
    print(
        f"{'ok  ' if pure_mf_ok else 'FAIL'} matrix-free GMRES variable scaling: "
        f"pressure diff {pure_mf_diff:.3e}, bhp diff {pure_mf_bhp:.3e}, "
        f"linear iterations {pure_mf.report[0].linear_iterations}"
    )
    failed += 0 if pure_mf_ok else 1

    try:
        import cupy  # noqa: F401
    except ImportError:
        print("skip cupy sparse Newton solve: cupy is not installed")
    else:
        cupy_run = run_backend_blackoil_tutorial_spe1(
            nsteps=1,
            verbose=False,
            linear_solver="cupy_sparse",
        )
        cupy_diff = _maxdiff(cupy_run.history["pressure"], dense_first.history["pressure"])
        cupy_ok = cupy_diff <= 1e-2
        print(f"{'ok  ' if cupy_ok else 'FAIL'} cupy sparse Newton solve: pressure diff {cupy_diff:.3e}")
        failed += 0 if cupy_ok else 1
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
