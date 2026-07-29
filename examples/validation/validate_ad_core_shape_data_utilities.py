"""Validate ``expandIfUniform``, ``expandMatrixToCell``, ``numelData`` and ``sizeJac``."""

from pathlib import Path

import numpy as np
import scipy.io
import scipy.sparse as sp

from pymrst.modules.ad_core.backends.diagonal import DiagonalJacobian
from pymrst.modules.ad_core.utils import (
    expand_if_uniform,
    expand_matrix_to_cell,
    numel_data,
    size_jac,
)


def main() -> int:
    ref = scipy.io.loadmat(
        Path(__file__).parent / "ref_ad_core_shape_data_utilities.mat",
        squeeze_me=True,
    )
    matrix = np.asarray(ref["matrix"])
    expanded = expand_if_uniform(matrix)
    subset = expand_matrix_to_cell(matrix, [0, 2])
    checks = {
        "expanded 1": (expanded[0], ref["e1"]),
        "expanded 2": (expanded[1], ref["e2"]),
        "expanded 3": (expanded[2], ref["e3"]),
        "subset 1": (subset[0], ref["s1"]),
        "subset 2": (subset[1], ref["s2"]),
        "subset 3": (subset[2], ref["s3"]),
        "single column": (expand_if_uniform(np.arange(4)[:, None]), ref["single"]),
        "numel matrix": (numel_data(np.zeros((2, 3))), ref["n_matrix"]),
        "numel list": (numel_data([1, 2, 3]), ref["n_cell"]),
        "numel scalar": (numel_data(4), ref["n_scalar"]),
        "sparse size": (size_jac(sp.csr_matrix((3, 7))), ref["size_sparse"]),
        "diagonal size": (size_jac(DiagonalJacobian.zeros(4, 12)), ref["size_diagonal"]),
        "diagonal columns": (size_jac(DiagonalJacobian.zeros(4, 12), 1), ref["size_diagonal_dim2"]),
    }
    failed = 0
    for name, (actual, expected) in checks.items():
        actual, expected = np.asarray(actual).squeeze(), np.asarray(expected).squeeze()
        ok = actual.shape == expected.shape and np.array_equal(actual, expected)
        print(f"{'ok  ' if ok else 'FAIL'} {name}")
        failed += not ok
    return int(failed != 0)


if __name__ == "__main__":
    raise SystemExit(main())
