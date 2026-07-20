"""Phase 92: MRST-style SR1 trust-region optimizer for calibration."""

import numpy as np

from pymrst.modules.vemmech.calibration import optimize_sr1


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
    A = np.array([[5.0, 1.0], [1.0, 2.0]])
    b = np.array([-2.0, 0.5])
    optimum = -np.linalg.solve(A, b)

    def fun(u):
        u = np.asarray(u, dtype=float).ravel()
        val = 0.5 * u @ A @ u + b @ u
        grad = A @ u + b
        return val, grad, None, None

    start = np.array([0.9, -0.8])
    fopt, uopt, history = optimize_sr1(
        fun,
        start,
        bounds=None,
        delta=0.2,
        grad_tol=1e-10,
        obj_change_tol=1e-14,
        max_it=30,
    )
    start_val = fun(start)[0]
    expected_val = fun(optimum)[0]

    failures = 0
    print("Phase 92: calibration SR1 trust-region optimizer")
    failures += compare("optimal controls", uopt, optimum, rtol=1e-6, atol=1e-8)
    failures += compare("optimal value", [fopt], [expected_val],
                        rtol=1e-9, atol=1e-10)
    if not fopt < start_val:
        print(f"  FAIL objective reduction: start={start_val:.3e}, opt={fopt:.3e}")
        failures += 1
    else:
        print(f"  ok   objective reduction: start={start_val:.3e}, opt={fopt:.3e}")
    failures += compare("history final value", [history.values[-1]], [fopt],
                        rtol=1e-12, atol=1e-12)
    failures += compare("history has trust radii", [len(history.trust_radii) > 0],
                        [1], rtol=0, atol=0)
    failures += compare("history has hessians", [len(history.hessians) > 0],
                        [1], rtol=0, atol=0)
    if failures:
        print(f"\nSR1 validation failed ({failures} checks).")
    else:
        print("\nPyMRST calibration SR1 optimizer solves the quadratic check.")
    return failures


if __name__ == "__main__":
    raise SystemExit(main())
