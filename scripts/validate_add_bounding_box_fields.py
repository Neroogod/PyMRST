"""Check `add_bounding_box_fields` against MRST (scripts/matlab/ref_bounding_box.mat).

Regenerate the reference from the MRST root with:

    matlab -batch "startup; run('.../scripts/matlab/dump_bounding_box_reference.m')"

Cell and face boxes are compared as *sets of rows*, not row by row: PyMRST and
MRST number faces differently on the same Cartesian grid, so an entry-wise
comparison would fail for a reason that has nothing to do with this function.
The multiset of boxes is invariant under renumbering and is what the function
actually promises.
"""

import sys
from pathlib import Path

import numpy as np
from scipy.io import loadmat

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pymrst.gridprocessing import cart_grid, compute_geometry        # noqa: E402
from pymrst.utils.gridtools.add_bounding_box_fields import (         # noqa: E402
    add_bounding_box_fields,
)

CASES = {"c322": ([3, 2, 2], [3.0, 2.0, 2.0]),
         "c33": ([3, 3], [3.0, 3.0]),
         "stretched": ([2, 2, 2], [4.0, 6.0, 10.0])}


def as_multiset(rows):
    rows = np.asarray(rows, dtype=float)
    return rows[np.lexsort(tuple(rows[:, d] for d in range(rows.shape[1])))]


def main() -> int:
    reference = loadmat(Path(__file__).parent / "matlab" / "ref_bounding_box.mat",
                        squeeze_me=True, struct_as_record=False)
    failures = []
    for name, (dims, sizes) in CASES.items():
        ref = reference[name]
        got = add_bounding_box_fields(
            compute_geometry(cart_grid(list(dims), list(sizes))),
            cells=True, faces=True)
        for field, mine, theirs in (
                ("cells.bbox", got.cells.bbox, ref.cellBbox),
                ("faces.bbox", got.faces.bbox, ref.faceBbox)):
            mine, theirs = np.asarray(mine, float), np.asarray(theirs, float)
            ok = (mine.shape == theirs.shape
                  and np.allclose(as_multiset(mine), as_multiset(theirs)))
            print(f"  {'ok ' if ok else 'FAIL'} {name}.{field}  "
                  f"{mine.shape} vs {theirs.shape}")
            if not ok:
                failures.append(f"{name}.{field}")

    print()
    if failures:
        print(f"FAILED: {', '.join(failures)}")
        return 1
    print("add_bounding_box_fields matches MRST's addBoundingBoxFields.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
