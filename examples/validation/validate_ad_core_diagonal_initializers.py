"""Direct MATLAB/MRST Jacobian validation for diagonal AD initializers."""

from pathlib import Path

import numpy as np
from scipy.io import loadmat

from pymrst.modules.ad_core.backends.diagonal.utils.init_variables_ad_diagonal import (
    initVariablesAD_diagonal,
)
from pymrst.modules.ad_core.backends.diagonal.utils.init_variables_ad_diagonal_row_major import (
    initVariablesAD_diagonalRowMajor,
)
from pymrst.modules.ad_core.backends.diagonal.utils.init_variables_ad_one_block import (
    initVariablesAD_oneBlock,
)


def _matrix(value):
    return value.toarray() if hasattr(value, "toarray") else np.asarray(value)


def main() -> int:
    ref = loadmat(
        Path(__file__).with_name("ref_ad_core_diagonal_initializers.mat")
    )
    a, b, c = np.array([1.0, 2.0]), np.array([3.0, 4.0]), np.array([5.0])
    diagonal = initVariablesAD_diagonal(a, b, c)
    row_major = initVariablesAD_diagonalRowMajor(a, b, c)
    one_block = initVariablesAD_oneBlock(a, c, b)
    typed = initVariablesAD_diagonal(a, c, b, types=[1, 2, 1])

    comparisons = {}
    for prefix, variables in (
        ("jd", diagonal),
        ("jr", row_major),
        ("jt", typed),
    ):
        for vi, variable in enumerate(variables, 1):
            for ji, block in enumerate(variable.jac, 1):
                comparisons[f"{prefix}{vi}{ji}"] = (
                    block.to_sparse().toarray(),
                    _matrix(ref[f"{prefix}{vi}{ji}"]),
                )
    for vi, variable in enumerate(one_block, 1):
        comparisons[f"jo{vi}"] = (
            variable.jac[0].to_sparse().toarray(),
            _matrix(ref[f"jo{vi}"]),
        )

    failed = 0
    print("Diagonal AD initializer Jacobians vs MATLAB/MRST:")
    for name, (actual, expected) in comparisons.items():
        same_shape = actual.shape == expected.shape
        error = (
            np.max(np.abs(actual - expected))
            if same_shape and actual.size
            else (0.0 if same_shape else np.inf)
        )
        ok = same_shape and error < 1e-14
        failed += not ok
        print(
            f"  {'ok  ' if ok else 'FAIL'} {name}: "
            f"shape {actual.shape}, max|dJ| = {error:.3e}"
        )
    print("\nInitializer Jacobians match MATLAB/MRST!" if not failed
          else f"\n{failed} comparisons FAILED")
    return int(bool(failed))


if __name__ == "__main__":
    raise SystemExit(main())
