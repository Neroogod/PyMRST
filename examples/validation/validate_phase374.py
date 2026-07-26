"""Phase 374: refineGrdecl against MRST (Phase 25 core leftovers).

Pins `refine_grdecl`/`grdecl_xyz` against MRST's `refineGrdecl` on a non-trivial
wavy corner-point deck refined by [2 2 2]. PyMRST refines the identical dumped
input and compares the refined cartDims, COORD, ZCORN and inherited PERMX number
for number.
"""

from pathlib import Path

import numpy as np
from scipy.io import loadmat

from pymrst.gridprocessing.refine_grdecl import refine_grdecl

REF = Path(__file__).with_name("ref_p374_refine.mat")


def _check(name: str, ok: bool, detail: str = "") -> int:
    print(f"  {'ok  ' if ok else 'FAIL'} {name}{('  ' + detail) if detail else ''}")
    return 0 if ok else 1


def main() -> int:
    if not REF.exists():
        print(f"missing {REF.name}; run dump_phase374_reference.m in MATLAB first")
        return 1
    ref = loadmat(REF)

    grdecl_in = {
        "cartDims": ref["in_cartDims"].ravel().astype(np.int64),
        "COORD": ref["in_COORD"].ravel(),
        "ZCORN": ref["in_ZCORN"].ravel(),
        "PERMX": ref["in_PERMX"].ravel(),
    }
    dim = ref["dim"].ravel().astype(int)

    out = refine_grdecl(grdecl_in, dim)

    failed = 0
    print(f"refineGrdecl (wavy 3x2x2, factors {list(dim)}) vs MRST:")
    failed += _check("cartDims match",
                     np.array_equal(out["cartDims"], ref["out_cartDims"].ravel()))
    err_c = np.max(np.abs(out["COORD"] - ref["out_COORD"].ravel()))
    failed += _check("COORD matches", err_c < 1e-9, f"max|dC| = {err_c:.3e}")
    err_z = np.max(np.abs(out["ZCORN"] - ref["out_ZCORN"].ravel()))
    failed += _check("ZCORN matches", err_z < 1e-9, f"max|dZ| = {err_z:.3e}")
    err_p = np.max(np.abs(out["PERMX"] - ref["out_PERMX"].ravel()))
    failed += _check("inherited PERMX matches", err_p < 1e-12,
                     f"max|dP| = {err_p:.3e}")

    print("\nrefineGrdecl matches MRST!" if failed == 0
          else f"\n{failed} checks FAILED")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
