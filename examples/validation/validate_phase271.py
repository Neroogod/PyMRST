"""Phase 271: expandMatrixToCell utility parity."""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pymrst.modules.compositional.utils import expandMatrixToCell, expand_matrix_to_cell, list_utils  # noqa: E402


def _check(name: str, condition: bool) -> int:
    print(f"  {'ok  ' if condition else 'FAIL'} {name}")
    return 0 if condition else 1


def main() -> int:
    matrix = np.arange(12).reshape(4, 3)
    cols = expandMatrixToCell(matrix)
    subset_cols = expand_matrix_to_cell(matrix, subset=[0, 2])
    colon_cols = expandMatrixToCell(matrix, subset=":")
    vector_cols = expandMatrixToCell(np.array([4.0, 5.0, 6.0]))
    existing = expandMatrixToCell([np.array([1, 2]), np.array([3, 4])])
    catalog = list_utils()

    print("\nPhase 271: expandMatrixToCell utility parity")
    failures = 0
    failures += _check("2-D matrix splits into column vectors",
                       len(cols) == 3
                       and all(np.array_equal(cols[i], matrix[:, i]) for i in range(3)))
    failures += _check("row subset is applied to each column",
                       len(subset_cols) == 3
                       and all(np.array_equal(subset_cols[i], matrix[[0, 2], i]) for i in range(3)))
    failures += _check("colon subset matches full slice",
                       all(np.array_equal(colon_cols[i], matrix[:, i]) for i in range(3)))
    failures += _check("1-D vector is treated as one column",
                       len(vector_cols) == 1 and np.array_equal(vector_cols[0], [4.0, 5.0, 6.0]))
    failures += _check("cell/list input passes through as list",
                       len(existing) == 2 and np.array_equal(existing[0], [1, 2]) and np.array_equal(existing[1], [3, 4]))
    failures += _check("utility catalog marks expandMatrixToCell implemented",
                       catalog["expandMatrixToCell"]["status"] == "implemented"
                       and catalog["expandMatrixToCell"]["python_module"] == "expand_matrix_to_cell")

    if failures:
        print(f"\nPhase 271 failed ({failures} checks).")
        return 1
    print("\nexpandMatrixToCell now mirrors MRST column-to-cell expansion semantics.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
