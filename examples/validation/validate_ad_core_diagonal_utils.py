"""Direct MATLAB/MRST validation of portable diagonal backend utilities."""

from pathlib import Path

import numpy as np
import scipy.sparse as sp
from scipy.io import loadmat

from pymrst.modules.ad_core.backends.diagonal.utils import (
    diagProductMult,
    getSparseBlocks,
    incrementSubset,
)


def main() -> int:
    ref = loadmat(Path(__file__).with_name("ref_ad_core_diagonal_utils.mat"))
    v1, v2 = ref["v1"].ravel(), ref["v2"].ravel()
    x, y = sp.csr_array(ref["x"]), sp.csr_array(ref["y"])
    product, _, _ = diagProductMult(v1, v2, x, y)
    incremented = incrementSubset(
        ref["base"].ravel(), ref["subset"].ravel().astype(int) - 1, 5.0
    )
    row, col, data = getSparseBlocks(x, ioffset=2, joffset=4)
    py_sparse = sp.csr_array((data, (row, col)), shape=(5, 6))
    ml_sparse = sp.csr_array(
        (
            ref["data"].ravel(),
            (
                ref["row"].ravel().astype(int) - 1,
                ref["col"].ravel().astype(int) - 1,
            ),
        ),
        shape=(5, 6),
    )

    comparisons = {
        "diagProductMult": (product.toarray(), ref["product"].toarray()),
        "incrementSubset": (incremented, ref["incremented"]),
        "getSparseBlocks": (py_sparse.toarray(), ml_sparse.toarray()),
    }
    failed = 0
    print("Diagonal backend utilities vs MATLAB/MRST:")
    for name, (actual, expected) in comparisons.items():
        error = np.max(np.abs(np.asarray(actual).ravel() - np.asarray(expected).ravel()))
        ok = error < 1e-14
        failed += not ok
        print(f"  {'ok  ' if ok else 'FAIL'} {name}: max|d| = {error:.3e}")
    return int(bool(failed))


if __name__ == "__main__":
    raise SystemExit(main())
