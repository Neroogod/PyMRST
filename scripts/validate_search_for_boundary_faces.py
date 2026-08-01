"""Check `search_for_boundary_faces` against MRST (scripts/matlab/ref_search_boundary.mat).

Regenerate the reference from the MRST root with:

    matlab -batch "startup; run('.../scripts/matlab/dump_search_boundary_reference.m')"

Faces are compared by **centroid**, not by index: MRST and PyMRST number the
faces of the same `cartGrid` differently, so the index sets would disagree
while naming the same faces. A centroid identifies a face in either numbering,
and the sorted arrays must match exactly.

The restricted cases carry the index-base conversion, which is the thing most
likely to be wrong: MRST's `i3 = 1` is PyMRST's `i3 = [0]`. The `i1 = 2` case
is included because it must come back **empty** - west faces all have `I = 0`,
so restricting to `I = 1` selects nothing - and a validator that only compared
non-empty results would not notice an off-by-one there.
"""

import sys
from pathlib import Path

import numpy as np
from scipy.io import loadmat

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pymrst.gridprocessing import cart_grid, compute_geometry               # noqa: E402
from pymrst.utils.gridtools.search_for_boundary_faces import (              # noqa: E402
    search_for_boundary_faces,
)


def sorted_rows(a):
    a = np.atleast_2d(np.asarray(a, dtype=float))
    if a.size == 0:
        return a.reshape(0, 3)
    return a[np.lexsort(tuple(a[:, d] for d in range(a.shape[1])))]


def main() -> int:
    reference = loadmat(Path(__file__).parent / "matlab" / "ref_search_boundary.mat",
                        squeeze_me=True, struct_as_record=False)
    g = compute_geometry(cart_grid([4, 3, 2], [4.0, 3.0, 2.0]))
    g2 = compute_geometry(cart_grid([4, 3], [4.0, 3.0]))
    centroids = np.asarray(g.faces.centroids, dtype=float)
    centroids2 = np.asarray(g2.faces.centroids, dtype=float)

    cases = [(f"s_{side}", centroids[search_for_boundary_faces(g, side)])
             for side in ("west", "east", "south", "north", "top", "bottom")]
    cases += [
        ("r_k1", centroids[search_for_boundary_faces(g, "west", i3=[0])]),
        ("r_i1", centroids[search_for_boundary_faces(g, "west", i1=[0])]),
        ("r_i2", centroids[search_for_boundary_faces(g, "west", i1=[1])]),
        ("r_combo", centroids[search_for_boundary_faces(g, "top", i1=[0, 1],
                                                        i2=[2])]),
        ("t2d_west", centroids2[search_for_boundary_faces(g2, "west")]),
        ("t2d_north", centroids2[search_for_boundary_faces(g2, "north")]),
    ]

    failures = []
    for name, mine in cases:
        theirs = np.asarray(reference[name], dtype=float)
        a, b = sorted_rows(mine), sorted_rows(theirs)
        ok = a.shape == b.shape and (a.size == 0 or np.allclose(a, b))
        note = " (empty, as expected)" if b.size == 0 else ""
        print(f"  {'ok ' if ok else 'FAIL'} {name}: {a.shape[0]} faces "
              f"vs {b.shape[0]}{note}")
        if not ok:
            failures.append(name)

    print()
    if failures:
        print(f"FAILED: {', '.join(failures)}")
        return 1
    print("search_for_boundary_faces matches MRST's searchForBoundaryFaces.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
