"""Phase 367: makeInternalBoundary against MRST (Phase 25 core leftovers).

Pins `make_internal_boundary` against MRST's `makeInternalBoundary` on a 3x3x2
grid cut along the same deterministic set of internal faces. Face renumbering
after remove/add differs between the two codes, so the check uses
renumbering-independent quantities that still fully pin the resulting topology:
the face and cell counts, the boundary-face count, and the sorted multiset of
neighbour pairs (outside encoded as -1).
"""

from pathlib import Path

import numpy as np
from scipy.io import loadmat

from pymrst.gridprocessing import cart_grid, compute_geometry
from pymrst.gridprocessing.make_internal_boundary import make_internal_boundary

REF = Path(__file__).with_name("ref_p367_intbnd.mat")


def _check(name: str, ok: bool, detail: str = "") -> int:
    print(f"  {'ok  ' if ok else 'FAIL'} {name}{('  ' + detail) if detail else ''}")
    return 0 if ok else 1


def _sorted_pairs(neigh: np.ndarray) -> np.ndarray:
    nb = np.sort(neigh, axis=1)
    return nb[np.lexsort((nb[:, 1], nb[:, 0]))]


def main() -> int:
    if not REF.exists():
        print(f"missing {REF.name}; run dump_phase367_reference.m in MATLAB first")
        return 1
    ref = loadmat(REF)
    cut = ref["cut0"].ravel().astype(np.int64)

    g = compute_geometry(cart_grid([3, 3, 2], [3.0, 3.0, 2.0]))
    g2, N = make_internal_boundary(g, cut)

    failed = 0
    print("makeInternalBoundary vs MRST (3x3x2, every-other internal face):")
    failed += _check(f"face count ({g2.faces.num})",
                     g2.faces.num == int(ref["faces_num"].ravel()[0]))
    failed += _check(f"cell count ({g2.cells.num})",
                     g2.cells.num == int(ref["cells_num"].ravel()[0]))
    neigh = np.asarray(g2.faces.neighbors, dtype=np.int64)
    n_boundary = int(np.any(neigh == -1, axis=1).sum())
    failed += _check(f"boundary-face count ({n_boundary})",
                     n_boundary == int(ref["n_boundary"].ravel()[0]))

    pairs = _sorted_pairs(neigh)
    # MRST cells are 1-based (outside already encoded as -1); shift real cells
    # to PyMRST's 0-based numbering before comparing.
    ref_pairs = np.asarray(ref["pairs"], dtype=np.int64).copy()
    ref_pairs[ref_pairs != -1] -= 1
    pairs_ref = _sorted_pairs(ref_pairs)
    failed += _check("neighbour-pair multiset matches (topology identical)",
                     pairs.shape == pairs_ref.shape and np.array_equal(pairs, pairs_ref))
    failed += _check("every cut face returned two replacements",
                     np.all(np.isfinite(N)) and N.shape[0] == cut.size)

    print("\nmakeInternalBoundary matches MRST!" if failed == 0
          else f"\n{failed} checks FAILED")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
