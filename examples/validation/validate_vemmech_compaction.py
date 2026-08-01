"""Validate 2-D compaction against MRST, on plain and triangulated grids.

The triangulated case is the one that matters. Both PyMRST and MRST must
reproduce the analytic column solution there, and an earlier defect - cell-face
tags losing their alignment when edges were sorted - showed up as a factor of
2.2 in exactly this comparison while leaving the plain grid perfect.
"""

from pathlib import Path

import numpy as np
from scipy.io import loadmat

from pymrst.modules.vemmech.examples.compaction_test_2d import run_compaction_test_2d

REF = Path(__file__).with_name("ref_vemmech_compaction.mat")


def _check(name: str, ok: bool, detail: str = "") -> int:
    print(f"  {'ok  ' if ok else 'FAIL'} {name}{('  ' + detail) if detail else ''}")
    return 0 if ok else 1


def _close(actual, expected, tol=1e-9):
    actual = np.asarray(actual, dtype=float).ravel()
    expected = np.asarray(expected, dtype=float).ravel()
    if actual.shape != expected.shape:
        return False, f"shape {actual.shape} vs {expected.shape}"
    scale = max(float(np.abs(expected).max()), 1.0)
    error = float(np.abs(actual - expected).max()) / scale
    return error < tol, f"max relative |d| = {error:.3e}"


def main() -> int:
    if not REF.exists():
        print(f"missing {REF.name}; run dump_vemmech_compaction_reference.m")
        return 1
    ref = loadmat(REF, squeeze_me=True)
    failed = 0

    for label, triangulate, prefix in (("plain grid", False, "plain"),
                                       ("triangulated grid", True, "tri")):
        print(f"2D compaction, {label}:")
        result = run_compaction_test_2d(cart_dims=[10, 10], L=[15.0, 15.0],
                                        triangulate=triangulate, twist=False)
        dim = result.grid.griddim
        computed = result.u[:, dim - 1]

        failed += _check(f"cell count ({result.grid.cells.num})",
                         result.grid.cells.num == int(ref[f"{prefix}_cells"]))
        failed += _check("minimum vertical displacement",
                         *_close(computed.min(), ref[f"{prefix}_uz_min"]))
        failed += _check("analytic profile agrees with MRST's",
                         *_close(result.analytic.min(), ref[f"{prefix}_ana_min"]))
        # MRST orders nodes the same way, so this compares node for node
        failed += _check("full displacement field",
                         *_close(computed, ref[f"{prefix}_uz"]))
        ratio = computed.min() / result.analytic.min()
        failed += _check(f"matches the analytic column ({ratio:.6f})",
                         abs(ratio - 1.0) < 1e-9)
    return int(failed > 0)


if __name__ == "__main__":
    raise SystemExit(main())
