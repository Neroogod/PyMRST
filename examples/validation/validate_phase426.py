"""Phase 426: compare computeCpGeometry with MRST on a tilted deck."""

from pathlib import Path

import numpy as np
from scipy.io import loadmat

from pymrst.gridprocessing import process_grdecl
from pymrst.modules.ad_core.utils import compute_cp_geometry

REF = Path(__file__).with_name("ref_p426_cp_geometry.mat")


def _check(name, actual, expected, tolerance=1e-11):
    actual = np.asarray(actual)
    expected = np.asarray(expected)
    error = np.max(np.abs(actual - expected)) if actual.size else 0.0
    ok = actual.shape == expected.shape and error < tolerance
    print(f"  {'ok  ' if ok else 'FAIL'} {name}: shape={actual.shape}, max|d|={error:.3e}")
    return not ok


def main():
    if not REF.exists():
        print(f"missing {REF.name}; run dump_phase426_reference.m in MATLAB first")
        return 1
    ref = loadmat(REF)
    grdecl = {
        "cartDims": ref["in_cartDims"].ravel().astype(np.int64),
        "COORD": ref["in_COORD"].ravel(),
        "ZCORN": ref["in_ZCORN"].ravel(),
        "ACTNUM": ref["in_ACTNUM"].ravel().astype(np.int64),
    }
    grid = process_grdecl(grdecl)
    active = compute_cp_geometry(grid, grdecl)
    all_cpc = compute_cp_geometry(grid, grdecl, on_all_cpc=True)

    failed = 0
    print("computeCpGeometry vs MRST:")
    for name, actual, expected in (
        ("active cell centers", active[0], ref["out_active_cc"]),
        ("active face centers", active[1], ref["out_active_fc"]),
        ("active-call dimensions", active[2], ref["out_dims"]),
        ("all cell centers", all_cpc[0], ref["out_all_cc"]),
        ("all face centers", all_cpc[1], ref["out_all_fc"]),
        ("all-call dimensions", all_cpc[2], ref["out_all_dims"]),
    ):
        failed += _check(name, actual, expected)

    print("\ncomputeCpGeometry matches MRST!" if not failed
          else f"\n{failed} checks FAILED")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
