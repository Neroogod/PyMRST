"""Phase 304: diagonal backend Numba-kernel report and row-major stacking."""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import scipy.sparse as sp

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pymrst.modules.ad_core.backends.diagonal import (  # noqa: E402
    DiagonalAutoDiffBackend,
    DiagonalJacobian,
    RowMajorDiagonalJacobian,
    combine_diagonal_equations,
    init_variables_diagonal,
)
from pymrst.modules.ad_core.backends.diagonal.kernels import HAS_NUMBA_KERNELS  # noqa: E402


def _check(name: str, condition: bool) -> int:
    print(f"  {'ok  ' if condition else 'FAIL'} {name}")
    return 0 if condition else 1


def validate_kernel_report() -> int:
    backend = DiagonalAutoDiffBackend(use_mex=True, row_major=True)
    report = backend.kernel_report()
    alias = backend.kernelReport()
    required = {
        "add_row_major",
        "left_scale_row_major",
        "matmul_left_row_major",
        "matmul_left_triplet",
        "row_repeat_row_major",
        "row_repeat_triplet",
        "take_rows_row_major",
        "take_rows_triplet",
        "vstack_row_major",
        "vstack_triplet",
    }
    failures = 0
    failures += _check("kernel report exposes MRST MEX-labelled backend row",
                       report["backend"] == "Diagonal (MEX+RowMajor)"
                       and report["use_mex"] is True
                       and report["row_major"] is True)
    failures += _check("kernel report lists every compact operation",
                       set(report["active_kernels"]) == required)
    failures += _check("kernel report matches installed Numba availability",
                       report["numba_available"] is HAS_NUMBA_KERNELS
                       and report["all_requested_kernels_active"] is HAS_NUMBA_KERNELS)
    failures += _check("camelCase kernelReport alias is identical",
                       alias == report)
    return failures


def validate_triplet_and_row_major_kernel_equivalence() -> int:
    mat = sp.csr_array(
        [
            [2.0, 0.0, -1.0],
            [0.0, 0.5, 1.0],
        ]
    )
    dense = sp.csr_array(
        [
            [1.0, 2.0, 0.0],
            [0.0, 3.0, 4.0],
            [5.0, 0.0, 6.0],
        ]
    )
    triplet = DiagonalJacobian.from_sparse(dense)
    row_major = RowMajorDiagonalJacobian.from_sparse(dense)

    failures = 0
    failures += _check("triplet matmul-left matches sparse product",
                       np.allclose(triplet.matmul_left(mat).toarray(), (mat @ dense).toarray()))
    failures += _check("row-major matmul-left matches sparse product",
                       np.allclose(row_major.matmul_left(mat).toarray(), (mat @ dense).toarray()))
    failures += _check("triplet row selection/repeat matches sparse references",
                       np.allclose(triplet.take_rows([2, 0]).toarray(), dense[[2, 0], :].toarray())
                       and np.allclose(triplet.take_rows([0]).row_repeat(3).toarray(),
                                       np.repeat(dense[[0], :].toarray(), 3, axis=0)))
    failures += _check("row-major row selection/repeat matches sparse references",
                       np.allclose(row_major.take_rows([2, 0]).toarray(), dense[[2, 0], :].toarray())
                       and np.allclose(row_major.take_rows([0]).row_repeat(3).toarray(),
                                       np.repeat(dense[[0], :].toarray(), 3, axis=0)))
    failures += _check("row-major addition preserves sparse sum",
                       np.allclose((row_major + row_major).toarray(), (dense + dense).toarray()))
    return failures


def validate_row_major_constant_equation_stack() -> int:
    (x,) = init_variables_diagonal(np.array([1.0, 2.0]), row_major=True)
    eq = combine_diagonal_equations(x * 3.0, np.array([7.0]))
    failures = 0
    failures += _check("mixed row-major/constant equation values stack",
                       np.allclose(eq.val, [3.0, 6.0, 7.0]))
    failures += _check("constant row keeps row-major compact stack before sparse handoff",
                       np.allclose(eq.jac[0].toarray(), [[3.0, 0.0], [0.0, 3.0], [0.0, 0.0]]))
    return failures


def main() -> int:
    print("\nPhase 304: diagonal backend Numba kernels and row-major stacks")
    failures = validate_kernel_report()
    failures += validate_triplet_and_row_major_kernel_equivalence()
    failures += validate_row_major_constant_equation_stack()
    if failures:
        print(f"\nPhase 304 failed ({failures} checks).")
        return 1
    print("\nDiagonal backend kernel status and row-major equation stacking are validated.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
