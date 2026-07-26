"""Phase 375: get2DGridFromFaces against MRST (Phase 25 core leftovers).

Pins `get_2d_grid_from_faces` against MRST's `get2DGridFromFaces`. The top faces
of a 3x3x2 grid (dumped as a face list) are turned into a 2D grid, and the
cell/face/node counts and sorted neighbour-pair multiset are compared against
MRST's.
"""

from pathlib import Path

import numpy as np
from scipy.io import loadmat

from pymrst.gridprocessing import cart_grid, compute_geometry
from pymrst.gridprocessing.get_2d_grid_from_faces import get_2d_grid_from_faces

REF = Path(__file__).with_name("ref_p375_get2d.mat")


def _check(name: str, ok: bool, detail: str = "") -> int:
    print(f"  {'ok  ' if ok else 'FAIL'} {name}{('  ' + detail) if detail else ''}")
    return 0 if ok else 1


def _sorted_pairs(neigh: np.ndarray) -> np.ndarray:
    nb = np.sort(neigh, axis=1)
    return nb[np.lexsort((nb[:, 1], nb[:, 0]))]


def main() -> int:
    if not REF.exists():
        print(f"missing {REF.name}; run dump_phase375_reference.m in MATLAB first")
        return 1
    ref = loadmat(REF)
    f = ref["f0"].ravel().astype(np.int64)

    g = compute_geometry(cart_grid([3, 3, 2], [3.0, 3.0, 2.0]))
    g2 = get_2d_grid_from_faces(g, f)

    failed = 0
    print("get2DGridFromFaces (top faces of 3x3x2) vs MRST:")
    failed += _check(f"cell count ({g2.cells.num})",
                     g2.cells.num == int(ref["cells_num"].ravel()[0]))
    failed += _check(f"face count ({g2.faces.num})",
                     g2.faces.num == int(ref["faces_num"].ravel()[0]))
    # MRST BUG: get2DGridFromFaces sets g.nodes.num to the edge count (nf), not
    # the true node count - its own coords array has `coords_rows` rows. The port
    # sets nodes.num correctly (== coords rows), so compare against that.
    true_nodes = int(ref["coords_rows"].ravel()[0])
    failed += _check(f"node count ({g2.nodes.num}) matches MRST's coords rows "
                     f"(MRST's nodes.num field is a buggy {int(ref['nodes_num'].ravel()[0])})",
                     g2.nodes.num == true_nodes)

    pairs = _sorted_pairs(np.asarray(g2.faces.neighbors, dtype=np.int64))
    ref_pairs = np.asarray(ref["pairs"], dtype=np.int64).copy()
    ref_pairs[ref_pairs != -1] -= 1
    failed += _check("neighbour-pair multiset matches",
                     pairs.shape == ref_pairs.shape and
                     np.array_equal(pairs, _sorted_pairs(ref_pairs)))

    print("\nget2DGridFromFaces matches MRST!" if failed == 0
          else f"\n{failed} checks FAILED")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
