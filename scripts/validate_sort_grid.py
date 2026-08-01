"""Check `sort_grid` against MRST's `sortGrid` (scripts/matlab/ref_sort_grid.mat).

Regenerate the reference with, from the MRST root:

    matlab -batch "startup; run('.../scripts/matlab/dump_sort_grid_reference.m')"

The node, face and cell *ordering* must agree exactly. Two fields are expected
to differ, both documented in `sort_grid`'s module docstring:

- `faces.neighbors` in 2-D, where MRST leaves the columns unswapped after the
  node rotation reverses the face;
- `cells.faces`' tag column, which MRST's own source marks buggy.

The tag column is not dumped, so only the first divergence shows up here, and
this asserts on it rather than glossing over it.
"""

import sys
from pathlib import Path

import numpy as np
from scipy.io import loadmat

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pymrst.gridprocessing import cart_grid                     # noqa: E402
from pymrst.utils.gridtools.sort_grid import sort_grid          # noqa: E402

CASES = {"c33": ([3, 3], [3.0, 3.0]), "c43": ([4, 3], [4.0, 3.0]),
         "c222": ([2, 2, 2], [2.0, 2.0, 2.0]),
         "c322": ([3, 2, 2], [3.0, 2.0, 2.0])}


def main() -> int:
    reference = loadmat(Path(__file__).parent / "matlab" / "ref_sort_grid.mat",
                        squeeze_me=True, struct_as_record=False)
    failures = []
    for name, (dims, sizes) in CASES.items():
        ref = reference[name]
        got = sort_grid(cart_grid(list(dims), list(sizes)))
        two_d = len(dims) == 2

        checks = [
            ("nodes.coords", np.asarray(got.nodes.coords, dtype=float),
             np.asarray(ref.coords, dtype=float)),
            # MATLAB is 1-based throughout.
            ("faces.nodes", np.asarray(got.faces.nodes, dtype=np.int64),
             np.asarray(ref.faceNodes, dtype=np.int64).ravel() - 1),
            ("faces.nodePos", np.asarray(got.faces.nodePos, dtype=np.int64),
             np.asarray(ref.nodePos, dtype=np.int64).ravel() - 1),
            ("cells.faces", np.asarray(got.cells.faces, dtype=np.int64).ravel(),
             np.asarray(ref.cellFaces, dtype=np.int64).ravel() - 1),
            ("cells.facePos", np.asarray(got.cells.facePos, dtype=np.int64),
             np.asarray(ref.facePos, dtype=np.int64).ravel() - 1),
        ]
        for field, mine, theirs in checks:
            ok = (np.allclose(mine, theirs) if mine.dtype.kind == "f"
                  else np.array_equal(mine, theirs))
            print(f"  {'ok ' if ok else 'FAIL'} {name}.{field}")
            if not ok:
                failures.append(f"{name}.{field}")

        # MRST stores 0 for "no neighbour"; PyMRST stores -1.
        theirs = np.asarray(ref.neighbors, dtype=np.int64) - 1
        mine = np.asarray(got.faces.neighbors, dtype=np.int64)
        if two_d:
            # The documented divergence: MRST does not swap the columns of a
            # reversed face, so the pairs match only up to a swap.
            same_set = np.array_equal(np.sort(mine, axis=1),
                                      np.sort(theirs, axis=1))
            swapped = int(np.count_nonzero(np.any(mine != theirs, axis=1)))
            print(f"  {'ok ' if same_set else 'FAIL'} {name}.faces.neighbors "
                  f"(as a set; {swapped} pairs swapped, as documented)")
            if not same_set:
                failures.append(f"{name}.faces.neighbors")
            if swapped == 0:
                failures.append(f"{name}.faces.neighbors (expected a swap)")
        else:
            ok = np.array_equal(mine, theirs)
            print(f"  {'ok ' if ok else 'FAIL'} {name}.faces.neighbors")
            if not ok:
                failures.append(f"{name}.faces.neighbors")

    print()
    if failures:
        print(f"FAILED: {', '.join(failures)}")
        return 1
    print("sort_grid matches MRST's sortGrid on all cases.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
