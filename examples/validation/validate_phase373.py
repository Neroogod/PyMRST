"""Phase 373: findMatchingFaces against MRST (Phase 25 core leftovers).

Pins `find_matching_faces` against MRST's `findMatchingFaces`. A 3x3x2 grid is
cut along the same internal faces (making coincident boundary-face pairs), then
the matching pairs are recovered. Face renumbering differs, so the check uses the
pair count and the sorted set of matched-pair midpoint coordinates.
"""

from pathlib import Path

import numpy as np
from scipy.io import loadmat

from pymrst.gridprocessing import cart_grid, compute_geometry
from pymrst.gridprocessing.find_matching_faces import find_matching_faces
from pymrst.gridprocessing.make_internal_boundary import make_internal_boundary

REF = Path(__file__).with_name("ref_p373_matchfaces.mat")


def _check(name: str, ok: bool, detail: str = "") -> int:
    print(f"  {'ok  ' if ok else 'FAIL'} {name}{('  ' + detail) if detail else ''}")
    return 0 if ok else 1


def main() -> int:
    if not REF.exists():
        print(f"missing {REF.name}; run dump_phase373_reference.m in MATLAB first")
        return 1
    ref = loadmat(REF)
    cut = ref["cut0"].ravel().astype(np.int64)

    g = compute_geometry(cart_grid([3, 3, 2], [3.0, 3.0, 2.0]))
    g2, _ = make_internal_boundary(g, cut)
    g2 = compute_geometry(g2)
    pairs = find_matching_faces(g2)

    failed = 0
    print("findMatchingFaces (3x3x2 after cut) vs MRST:")
    failed += _check(f"matched-pair count ({pairs.shape[0]})",
                     pairs.shape[0] == int(ref["num_pairs"].ravel()[0]))

    cent = np.asarray(g2.faces.centroids, dtype=float)
    mids = np.round((cent[pairs[:, 0]] + cent[pairs[:, 1]]) / 2, 10)
    mids = mids[np.lexsort(mids.T[::-1])]
    ref_mids = np.asarray(ref["mids"], dtype=float)
    err = np.max(np.abs(mids - ref_mids)) if mids.shape == ref_mids.shape else np.inf
    failed += _check("matched-pair midpoints match", err < 1e-9,
                     f"max|dx| = {err:.3e}")

    print("\nfindMatchingFaces matches MRST!" if failed == 0
          else f"\n{failed} checks FAILED")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
