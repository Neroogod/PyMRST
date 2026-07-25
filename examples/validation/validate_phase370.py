"""Phase 370: hexahedralGrid against MRST (Phase 25 core leftovers).

Pins `hexahedral_grid` against MRST's `hexahedralGrid` on a two-cube mesh built
from the identical connectivity `H0`/`P`. Face renumbering may differ, so the
check uses the face/cell/node counts and the sorted multiset of neighbour pairs
(outside as -1, MRST's 1-based cells shifted to 0-based) - which fully pin the
assembled topology.
"""

from pathlib import Path

import numpy as np
from scipy.io import loadmat

from pymrst.gridprocessing.hexahedral_grid import hexahedral_grid

REF = Path(__file__).with_name("ref_p370_hexgrid.mat")


def _check(name: str, ok: bool, detail: str = "") -> int:
    print(f"  {'ok  ' if ok else 'FAIL'} {name}{('  ' + detail) if detail else ''}")
    return 0 if ok else 1


def _sorted_pairs(neigh: np.ndarray) -> np.ndarray:
    nb = np.sort(neigh, axis=1)
    return nb[np.lexsort((nb[:, 1], nb[:, 0]))]


def main() -> int:
    if not REF.exists():
        print(f"missing {REF.name}; run dump_phase370_reference.m in MATLAB first")
        return 1
    ref = loadmat(REF)
    H0 = np.asarray(ref["H0"], dtype=np.int64)
    P = np.asarray(ref["P"], dtype=float)

    g = hexahedral_grid(H0, P)

    failed = 0
    print("hexahedralGrid (two stacked cubes) vs MRST:")
    failed += _check(f"cell count ({g.cells.num})",
                     g.cells.num == int(ref["cells_num"].ravel()[0]))
    failed += _check(f"face count ({g.faces.num})",
                     g.faces.num == int(ref["faces_num"].ravel()[0]))
    failed += _check(f"node count ({g.nodes.num})",
                     g.nodes.num == int(ref["nodes_num"].ravel()[0]))

    pairs = _sorted_pairs(np.asarray(g.faces.neighbors, dtype=np.int64))
    ref_pairs = np.asarray(ref["pairs"], dtype=np.int64).copy()
    ref_pairs[ref_pairs != -1] -= 1                 # MRST 1-based cells -> 0-based
    pairs_ref = _sorted_pairs(ref_pairs)
    failed += _check("neighbour-pair multiset matches (topology identical)",
                     pairs.shape == pairs_ref.shape and np.array_equal(pairs, pairs_ref))

    print("\nhexahedralGrid matches MRST!" if failed == 0
          else f"\n{failed} checks FAILED")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
