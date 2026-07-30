"""Validate ``criticalPointChop``, ``splitMatrixForReduction`` and ``getCellMajorReordering``."""

from pathlib import Path

import numpy as np
import scipy.io
import scipy.sparse as sp

from pymrst.modules.ad_core.utils import (
    critical_point_chop,
    get_cell_major_reordering,
    split_matrix_for_reduction,
)


def main() -> int:
    ref = scipy.io.loadmat(
        Path(__file__).parent / "ref_ad_core_linear_algebra_utilities.mat",
        squeeze_me=True,
    )
    x0, trial = np.atleast_1d(ref["x0"]), np.atleast_1d(ref["trial"])
    A, b = sp.csr_matrix(ref["A"]), np.asarray(ref["b"])
    sparse = split_matrix_for_reduction(A, b, 2, "sparse", do_factor=False)
    subset = split_matrix_for_reduction(A, b, 2, "subset", do_factor=False)
    checks = {
        "chop both": (critical_point_chop(x0, trial, 0, 1, "both"), ref["chop_both"]),
        "chop above/increasing": (
            critical_point_chop(x0, trial, 0, 1, "above", "increasing"),
            ref["chop_above_inc"],
        ),
        "chop below/decreasing": (
            critical_point_chop(x0, trial, 0, 1, "below", "decreasing"),
            ref["chop_below_dec"],
        ),
        "B": (sparse.B.toarray(), ref["B"]), "C": (sparse.C.toarray(), ref["C"]),
        "D": (sparse.D.toarray(), ref["D"]), "E": (sparse.E.toarray(), ref["E"]),
        "f": (sparse.f, ref["f"]), "h": (sparse.h, ref["h"]),
        "subset B": (subset.B.toarray(), ref["B_subset"]),
        "subset E": (subset.E.toarray(), ref["E_subset"]),
        "ordering": (get_cell_major_reordering(3, 2) + 1, ref["ordering"]),
        "custom ordering": (
            get_cell_major_reordering(
                3, 2, ndof=8, equation_ordering=[1, 0], cell_ordering=[2, 0, 1]
            ) + 1,
            ref["ordering_custom"],
        ),
    }
    failed = 0
    for name, (actual, expected) in checks.items():
        actual, expected = np.asarray(actual).squeeze(), np.asarray(expected).squeeze()
        error = float(np.max(np.abs(actual - expected)))
        ok = actual.shape == expected.shape and np.allclose(actual, expected, rtol=1e-12, atol=1e-12)
        print(f"{'ok  ' if ok else 'FAIL'} {name}: max abs diff {error:.3e}")
        failed += not ok
    return int(failed != 0)


if __name__ == "__main__":
    raise SystemExit(main())
