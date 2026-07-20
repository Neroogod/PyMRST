"""Phase 93: calibration SR1 backup/resume and progress hooks."""

from pathlib import Path

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
    A = np.array([[4.0, 0.5], [0.5, 3.0]])
    b = np.array([-1.0, 0.75])
    optimum = -np.linalg.solve(A, b)

    def fun(u):
        u = np.asarray(u, dtype=float).ravel()
        return 0.5 * u @ A @ u + b @ u, A @ u + b, None, None

    backup = Path(__file__).with_name("tmp_phase93_sr1_backup.pkl")
    if backup.exists():
        backup.unlink()
    calls_first: list[float] = []
    calls_second: list[float] = []

    optimize_sr1(
        fun,
        np.array([0.9, -0.9]),
        bounds=None,
        delta=0.15,
        max_it=1,
        backup_file=str(backup),
        resume_backup=False,
        progress_callback=lambda h: calls_first.append(h.values[-1]),
    )
    fopt, uopt, history = optimize_sr1(
        fun,
        np.array([9.0, 9.0]),
        bounds=None,
        delta=0.15,
        max_it=40,
        backup_file=str(backup),
        resume_backup=True,
        progress_callback=lambda h: calls_second.append(h.values[-1]),
        grad_tol=1e-10,
        obj_change_tol=1e-14,
    )
    if backup.exists():
        backup.unlink()

    failures = 0
    print("Phase 93: calibration SR1 backup/resume and progress hooks")
    failures += compare("backup callbacks first run", [len(calls_first) > 0], [1],
                        rtol=0, atol=0)
    failures += compare("resume callbacks second run", [len(calls_second) > 0], [1],
                        rtol=0, atol=0)
    failures += compare("resumed optimum controls", uopt, optimum,
                        rtol=1e-6, atol=1e-8)
    failures += compare("resumed final value", [fopt], [fun(optimum)[0]],
                        rtol=1e-9, atol=1e-10)
    failures += compare("history resumed", [len(history.values) > 1], [1],
                        rtol=0, atol=0)
    if failures:
        print(f"\nSR1 backup/resume validation failed ({failures} checks).")
    else:
        print("\nPyMRST calibration SR1 backup/resume and progress hooks work.")
    return failures


if __name__ == "__main__":
    raise SystemExit(main())
