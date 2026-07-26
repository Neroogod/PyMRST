"""Phase 384: sliceGrid node-snapped cut against MRST (Phase 25).

Pins the PyMRST `slice_grid` node-snapping path against MRST's `sliceGrid` for
the diagonal plane x=y through a 2x2x1 grid - the plane passes exactly through
the corner nodes, cutting the two on-diagonal cells while snapping to shared
corners. Cell count and sorted per-cell volumes are compared.
"""

from pathlib import Path

import numpy as np
from scipy.io import loadmat

from pymrst.gridprocessing import cart_grid, compute_geometry
from pymrst.gridprocessing.slice_grid import slice_grid

REF = Path(__file__).with_name("ref_p384_slicesnap.mat")


def _check(name: str, ok: bool, detail: str = "") -> int:
    print(f"  {'ok  ' if ok else 'FAIL'} {name}{('  ' + detail) if detail else ''}")
    return 0 if ok else 1


def main() -> int:
    if not REF.exists():
        print(f"missing {REF.name}; run dump_phase384_reference.m in MATLAB first")
        return 1
    ref = loadmat(REF)

    g = compute_geometry(cart_grid([2, 2, 1], [2.0, 2.0, 1.0]))
    s = compute_geometry(slice_grid(
        g, np.array([[-1.0, -1.0, 0.0], [3.0, 3.0, 0.0]]),
        cut_dir=np.array([0.0, 0.0, 1.0])))

    failed = 0
    print("sliceGrid (snapped x=y cut of 2x2x1) vs MRST:")
    failed += _check(f"cell count ({s.cells.num})",
                     s.cells.num == int(ref["cells_num"].ravel()[0]))
    failed += _check(f"total volume ({s.cells.volumes.sum():.4f})",
                     np.isclose(s.cells.volumes.sum(),
                                float(ref["total_vol"].ravel()[0])))
    vol = np.sort(np.asarray(s.cells.volumes))
    err = np.max(np.abs(vol - ref["vol_sorted"].ravel()))
    failed += _check("sorted per-cell volumes match", err < 1e-9,
                     f"max|dV| = {err:.3e}")

    print("\nsliceGrid node-snapped cut matches MRST!" if failed == 0
          else f"\n{failed} checks FAILED")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
