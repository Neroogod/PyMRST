"""Phase 383: sliceGrid cell-splitting against MRST (Phase 25).

Pins the PyMRST `slice_grid` clean-cut port against MRST's `sliceGrid` for a
vertical plane x=1.5 cutting a 3x2x2 grid (splitting the four middle-column
cells). Cell renumbering differs, so the check uses the cell count, the total
volume, and the sorted per-cell volumes - which pin the geometry of the split.

Only the clean planar cut (through cell interiors, no node snapping) is ported;
MRST's `topologicalCellSplit` handling of node-snapped / partial cuts is out of
scope.
"""

from pathlib import Path

import numpy as np
from scipy.io import loadmat

from pymrst.gridprocessing import cart_grid, compute_geometry
from pymrst.gridprocessing.slice_grid import slice_grid

REF = Path(__file__).with_name("ref_p383_slice.mat")


def _check(name: str, ok: bool, detail: str = "") -> int:
    print(f"  {'ok  ' if ok else 'FAIL'} {name}{('  ' + detail) if detail else ''}")
    return 0 if ok else 1


def main() -> int:
    if not REF.exists():
        print(f"missing {REF.name}; run dump_phase383_reference.m in MATLAB first")
        return 1
    ref = loadmat(REF)

    g = compute_geometry(cart_grid([3, 2, 2], [3.0, 2.0, 2.0]))
    s = compute_geometry(slice_grid(
        g, np.array([[1.5, -1.0, 0.0], [1.5, 3.0, 0.0]]),
        cut_dir=np.array([0.0, 0.0, 1.0])))

    failed = 0
    print("sliceGrid (3x2x2 cut at x=1.5) vs MRST:")
    failed += _check(f"cell count ({s.cells.num})",
                     s.cells.num == int(ref["cells_num"].ravel()[0]))
    failed += _check(f"total volume ({s.cells.volumes.sum():.4f})",
                     np.isclose(s.cells.volumes.sum(),
                                float(ref["total_vol"].ravel()[0])))
    vol = np.sort(np.asarray(s.cells.volumes))
    err = np.max(np.abs(vol - ref["vol_sorted"].ravel()))
    failed += _check("sorted per-cell volumes match", err < 1e-9,
                     f"max|dV| = {err:.3e}")

    print("\nsliceGrid cell-splitting matches MRST!" if failed == 0
          else f"\n{failed} checks FAILED")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
