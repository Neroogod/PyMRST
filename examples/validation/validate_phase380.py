"""Phase 380: cellNodes against MRST (Phase 25 core leftovers).

Pins `cell_nodes` against MRST's `cellNodes` on a 2x2x1 Cartesian grid. The full
[cell, local-vertex, node] corner table is compared number for number (MRST's
1-based cell/vertex/node indices converted to 0-based), confirming the
direction-tag binary-key corner identification matches.
"""

from pathlib import Path

import numpy as np
from scipy.io import loadmat

from pymrst.gridprocessing import cart_grid, compute_geometry
from pymrst.gridprocessing.cell_nodes import cell_nodes

REF = Path(__file__).with_name("ref_p380_cellnodes.mat")


def _check(name: str, ok: bool, detail: str = "") -> int:
    print(f"  {'ok  ' if ok else 'FAIL'} {name}{('  ' + detail) if detail else ''}")
    return 0 if ok else 1


def main() -> int:
    if not REF.exists():
        print(f"missing {REF.name}; run dump_phase380_reference.m in MATLAB first")
        return 1
    ref = loadmat(REF)
    s_ref = np.asarray(ref["s"], dtype=np.int64) - 1     # 1-based -> 0-based

    g = compute_geometry(cart_grid([2, 2, 1], [2.0, 2.0, 1.0]))
    s = cell_nodes(g)

    failed = 0
    print("cellNodes (2x2x1) vs MRST:")
    failed += _check(f"same number of corner rows ({s.shape[0]})",
                     s.shape[0] == s_ref.shape[0])
    failed += _check("full [cell, local-vertex, node] table matches",
                     s.shape == s_ref.shape and np.array_equal(s, s_ref))

    print("\ncellNodes matches MRST!" if failed == 0
          else f"\n{failed} checks FAILED")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
