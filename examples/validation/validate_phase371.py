"""Phase 371: tessellationGrid against MRST (Phase 25 core leftovers).

Pins `tessellation_grid` against MRST's `tessellationGrid` on a 2x2 quad
tessellation built from identical `p`/`t0`. Face renumbering may differ, so the
check uses the face/cell/node counts and the sorted neighbour-pair multiset
(outside as -1, MRST's 1-based cells shifted to 0-based) - which fully pin the
assembled 2D topology.
"""

from pathlib import Path

import numpy as np
from scipy.io import loadmat

from pymrst.gridprocessing.tessellation_grid import tessellation_grid

REF = Path(__file__).with_name("ref_p371_tessgrid.mat")


def _check(name: str, ok: bool, detail: str = "") -> int:
    print(f"  {'ok  ' if ok else 'FAIL'} {name}{('  ' + detail) if detail else ''}")
    return 0 if ok else 1


def _sorted_pairs(neigh: np.ndarray) -> np.ndarray:
    nb = np.sort(neigh, axis=1)
    return nb[np.lexsort((nb[:, 1], nb[:, 0]))]


def main() -> int:
    if not REF.exists():
        print(f"missing {REF.name}; run dump_phase371_reference.m in MATLAB first")
        return 1
    ref = loadmat(REF)
    p = np.asarray(ref["p"], dtype=float)
    t0 = np.asarray(ref["t0"], dtype=np.int64)

    g = tessellation_grid(p, t0)

    failed = 0
    print("tessellationGrid (2x2 quads) vs MRST:")
    failed += _check(f"cell count ({g.cells.num})",
                     g.cells.num == int(ref["cells_num"].ravel()[0]))
    failed += _check(f"face count ({g.faces.num})",
                     g.faces.num == int(ref["faces_num"].ravel()[0]))
    failed += _check(f"node count ({g.nodes.num})",
                     g.nodes.num == int(ref["nodes_num"].ravel()[0]))

    pairs = _sorted_pairs(np.asarray(g.faces.neighbors, dtype=np.int64))
    ref_pairs = np.asarray(ref["pairs"], dtype=np.int64).copy()
    ref_pairs[ref_pairs != -1] -= 1
    pairs_ref = _sorted_pairs(ref_pairs)
    failed += _check("neighbour-pair multiset matches (topology identical)",
                     pairs.shape == pairs_ref.shape and np.array_equal(pairs, pairs_ref))

    print("\ntessellationGrid matches MRST!" if failed == 0
          else f"\n{failed} checks FAILED")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
