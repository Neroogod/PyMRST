"""Validate matrix-free backend linear solve helpers."""

from __future__ import annotations

import numpy as np

from pymrst.backends import torch_jvp_newton_step, torch_matrix_free_gmres


def main() -> int:
    try:
        import torch
    except ImportError:
        print("skip backend matrix-free validation: torch is not installed")
        return 0

    a = torch.tensor(
        [[4.0, 1.0, 0.0], [2.0, 3.0, 1.0], [0.0, 1.0, 2.0]],
        dtype=torch.float64,
    )
    b = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64)
    expect = torch.linalg.solve(a, b)
    x, report = torch_matrix_free_gmres(
        lambda v: a @ v,
        b,
        tolerance=1e-12,
        max_iterations=6,
    )
    diff = float(torch.max(torch.abs(x - expect)).cpu())
    ok = report.converged and diff <= 1e-11
    print(f"{'ok  ' if ok else 'FAIL'} matrix-free GMRES: max abs diff {diff:.3e}")

    def residual(y):
        return a @ y - b

    dx, jrep = torch_jvp_newton_step(
        residual,
        torch.zeros_like(b),
        b,
        tolerance=1e-12,
        max_iterations=6,
    )
    jdiff = float(torch.max(torch.abs(dx - expect)).cpu())
    jok = jrep.converged and jdiff <= 1e-11
    print(f"{'ok  ' if jok else 'FAIL'} JVP Newton step: max abs diff {jdiff:.3e}")
    return 0 if ok and jok else 1


if __name__ == "__main__":
    raise SystemExit(main())
