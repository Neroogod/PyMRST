"""Phase 372: triangulateFaces against MRST (Phase 25 core leftovers).

Pins `triangulate_faces` against MRST's `triangulateFaces` on a 2x2x1 grid (all
quad faces triangulated). Face renumbering may differ, so the check uses the
face/node counts, the sorted neighbour-pair multiset (outside as -1, MRST's
1-based cells shifted to 0-based), and the sorted set of added centroid-node
coordinates - pinning both the topology and the geometry of the split.
"""

from pathlib import Path

import numpy as np
from scipy.io import loadmat

from pymrst.gridprocessing import cart_grid, compute_geometry
from pymrst.gridprocessing.triangulate_faces import triangulate_faces

REF = Path(__file__).with_name("ref_p372_trifaces.mat")


def _check(name: str, ok: bool, detail: str = "") -> int:
    print(f"  {'ok  ' if ok else 'FAIL'} {name}{('  ' + detail) if detail else ''}")
    return 0 if ok else 1


def _sorted_pairs(neigh: np.ndarray) -> np.ndarray:
    nb = np.sort(neigh, axis=1)
    return nb[np.lexsort((nb[:, 1], nb[:, 0]))]


def _sorted_rows(a: np.ndarray) -> np.ndarray:
    return a[np.lexsort(a.T[::-1])]


def main() -> int:
    if not REF.exists():
        print(f"missing {REF.name}; run dump_phase372_reference.m in MATLAB first")
        return 1
    ref = loadmat(REF)

    g = compute_geometry(cart_grid([2, 2, 1], [2.0, 2.0, 1.0]))
    nn0 = g.nodes.num
    g2 = triangulate_faces(g)

    failed = 0
    print("triangulateFaces (2x2x1, all quads) vs MRST:")
    failed += _check(f"face count ({g2.faces.num})",
                     g2.faces.num == int(ref["faces_num"].ravel()[0]))
    failed += _check(f"node count ({g2.nodes.num})",
                     g2.nodes.num == int(ref["nodes_num"].ravel()[0]))

    pairs = _sorted_pairs(np.asarray(g2.faces.neighbors, dtype=np.int64))
    ref_pairs = np.asarray(ref["pairs"], dtype=np.int64).copy()
    ref_pairs[ref_pairs != -1] -= 1
    failed += _check("neighbour-pair multiset matches",
                     np.array_equal(pairs, _sorted_pairs(ref_pairs)))

    new_nodes = np.asarray(g2.nodes.coords, dtype=float)[nn0:]
    err = np.max(np.abs(_sorted_rows(new_nodes) - ref["new_nodes"]))
    failed += _check("added centroid-node coordinates match", err < 1e-12,
                     f"max|dx| = {err:.3e}")

    print("\ntriangulateFaces matches MRST!" if failed == 0
          else f"\n{failed} checks FAILED")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
