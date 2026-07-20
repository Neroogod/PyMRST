"""Validate backend Torch black-oil Newton step against SPE1 ADI model."""

from __future__ import annotations

import numpy as np

from validate_backend_blackoil_residual import _build_spe1
from pymrst.modules.ad_blackoil import (
    BackendBOState,
    BackendBlackOilModel,
    initial_state_spe1,
)
from pymrst.params import NORM_GRAVITY


def main() -> int:
    try:
        import torch  # noqa: F401
    except ImportError:
        print("skip backend black-oil model: torch is not installed")
        return 0

    deck, ref_model, residual_model = _build_spe1()
    st_ref0 = initial_state_spe1(ref_model, deck)
    st_backend0 = BackendBOState.from_bo_state(st_ref0)
    backend_model = BackendBlackOilModel(
        ref_model.g,
        ref_model.rock,
        residual_model.fluid,
        ref_model.wells,
        gravity=NORM_GRAVITY,
        max_iterations=8,
        tolerance=1e-7,
    )

    dt = float(deck["tsteps"][0])
    st_backend, report = backend_model.step(st_backend0, dt)
    st_ref, ok_ref, its_ref = ref_model.step(st_ref0, dt)
    checks = [
        ("pressure", st_backend.pressure, st_ref.pressure, 1e-2),
        ("saturation", st_backend.s, st_ref.s, 1e-10),
        ("rs", st_backend.rs, st_ref.rs, 1e-7),
        ("qWs", st_backend.qWs, st_ref.qWs, 1e-12),
        ("qOs", st_backend.qOs, st_ref.qOs, 1e-12),
        ("qGs", st_backend.qGs, st_ref.qGs, 1e-12),
        ("bhp", st_backend.bhp, st_ref.bhp, 2e-2),
    ]

    failed = 0
    if not report.converged or not ok_ref:
        print(
            "FAIL backend/ref convergence: "
            f"backend={report.converged}, ref={ok_ref}, "
            f"backend its={report.iterations}, ref its={its_ref}"
        )
        failed += 1
    else:
        print(
            f"ok   backend/ref convergence: backend its {report.iterations}, "
            f"ref its {its_ref}"
        )
    for name, got, ref, tol in checks:
        diff = float(np.max(np.abs(np.asarray(got) - np.asarray(ref))))
        ok = diff <= tol
        print(f"{'ok  ' if ok else 'FAIL'} {name}: max abs diff {diff:.3e}")
        failed += 0 if ok else 1
    status_ok = np.array_equal(st_backend.status, st_ref.status)
    print(f"{'ok  ' if status_ok else 'FAIL'} status flags")
    failed += 0 if status_ok else 1
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
