"""Phase 418: DG reference cubature rules against MRST.

Confirms that each PyMRST cubature getter returns MRST's exact points and weights
for every supported precision `k` (via the get*CubaturePointsAndWeights tables),
including the precision-threshold lookup. The reference `.mat` is the verbatim
dump of MRST's getters.
"""

from pathlib import Path

import numpy as np
from scipy.io import loadmat

from pymrst.modules.dg.cubatures.reference_cubature import (
    get_cube_cubature,
    get_line_cubature,
    get_square_cubature,
    get_tetrahedron_cubature,
    get_triangle_cubature,
)

REF = Path(__file__).with_name("ref_dg_cubature_tables.mat")

GETTERS = {
    "line": get_line_cubature,
    "square": get_square_cubature,
    "triangle": get_triangle_cubature,
    "cube": get_cube_cubature,
    "tet": get_tetrahedron_cubature,
}


def _check(name, ok, detail=""):
    print(f"  {'ok  ' if ok else 'FAIL'} {name}{('  ' + detail) if detail else ''}")
    return 0 if ok else 1


def main() -> int:
    if not REF.exists():
        print(f"missing {REF.name}; run dump_dg_cubature_tables.m in MATLAB first")
        return 1
    ref = loadmat(REF)

    failed = 0
    print("DG reference cubature getters vs MRST:")
    for nm, getter in GETTERS.items():
        ks = np.asarray(ref[nm + "_ks"]).ravel()
        max_xe = max_we = 0.0
        for k in ks:
            x, w, n, _ = getter(int(k))
            x_ref = np.asarray(ref[f"{nm}_k{k}_x"], dtype=float)
            w_ref = np.asarray(ref[f"{nm}_k{k}_w"], dtype=float).ravel()
            max_xe = max(max_xe, np.max(np.abs(x - x_ref)))
            max_we = max(max_we, np.max(np.abs(w - w_ref)))
        failed += _check(f"{nm}: all {ks.size} precisions match MRST",
                         max_xe < 1e-12 and max_we < 1e-12,
                         f"max|dx| {max_xe:.1e}, max|dw| {max_we:.1e}")

    print("\nDG reference cubatures match MRST!" if failed == 0
          else f"\n{failed} checks FAILED")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
