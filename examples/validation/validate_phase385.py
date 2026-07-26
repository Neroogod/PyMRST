"""Phase 385: sliceGrid partial cut against MRST (Phase 25).

Pins the PyMRST `slice_grid` partial-cut path against MRST's `sliceGrid`. A
segment stopping at x=2.5 cuts cells 0 and 1 of a 3x1x1 grid but not cell 2 (the
segment ends inside it): MRST removes partial cuts, and so does the port. Cell
count and sorted per-cell volumes are compared.
"""

from pathlib import Path

import numpy as np
from scipy.io import loadmat

from pymrst.gridprocessing import cart_grid, compute_geometry
from pymrst.gridprocessing.slice_grid import slice_grid

REF = Path(__file__).with_name("ref_p385_slicepartial.mat")


def _check(name: str, ok: bool, detail: str = "") -> int:
    print(f"  {'ok  ' if ok else 'FAIL'} {name}{('  ' + detail) if detail else ''}")
    return 0 if ok else 1


def main() -> int:
    if not REF.exists():
        print(f"missing {REF.name}; run dump_phase385_reference.m in MATLAB first")
        return 1
    ref = loadmat(REF)

    g = compute_geometry(cart_grid([3, 1, 1], [3.0, 1.0, 1.0]))
    s = compute_geometry(slice_grid(
        g, np.array([[0.0, 0.5, 0.0], [2.5, 0.5, 0.0]]),
        cut_dir=np.array([0.0, 0.0, 1.0])))

    failed = 0
    print("sliceGrid (partial cut of 3x1x1, window x<=2.5) vs MRST:")
    failed += _check(f"cell count ({s.cells.num})",
                     s.cells.num == int(ref["cells_num"].ravel()[0]))
    failed += _check(f"total volume ({s.cells.volumes.sum():.4f})",
                     np.isclose(s.cells.volumes.sum(),
                                float(ref["total_vol"].ravel()[0])))
    vol = np.sort(np.asarray(s.cells.volumes))
    err = np.max(np.abs(vol - ref["vol_sorted"].ravel()))
    failed += _check("sorted per-cell volumes match (partial cut removed)",
                     err < 1e-9, f"max|dV| = {err:.3e}")

    print("\nsliceGrid partial cut matches MRST!" if failed == 0
          else f"\n{failed} checks FAILED")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
