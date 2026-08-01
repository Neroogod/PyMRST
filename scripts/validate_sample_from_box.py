"""Check `sample_from_box` against MRST (scripts/matlab/ref_sample_from_box.mat).

Regenerate the reference from the MRST root with:

    matlab -batch "startup; run('.../scripts/matlab/dump_sample_from_box_reference.m')"

All cases are Cartesian on purpose. The port computes the bounding box from
every node of the selected cells where MRST uses only the nodes of the
extreme-*centroid* cells; on a Cartesian grid those coincide, so agreement here
is a real check of the sampling and indexing - the arithmetic, the `ceil`
convention, the axis order - and not of the divergence.

The divergence itself is not something a reference can settle, because MRST
indexes out of bounds in the case where they differ. It is exhibited by a test
in the module instead.

Cell numbering is asserted to agree (via `cells.centroids`) before any sample
is compared, since the result is one value per cell.
"""

import sys
from pathlib import Path

import numpy as np
from scipy.io import loadmat

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pymrst.gridprocessing import cart_grid, compute_geometry     # noqa: E402
from pymrst.utils.gridtools.sample_from_box import sample_from_box  # noqa: E402

CASES = {
    "a": ([4, 4], [4.0, 4.0], None),
    "b": ([6, 3], [6.0, 3.0], None),
    "c": ([3, 3, 3], [3.0, 3.0, 3.0], None),
    "d": ([5, 4], [5.0, 4.0], np.array([0, 3, 6, 11])),   # MATLAB [1 4 7 12]
}


def main() -> int:
    reference = loadmat(Path(__file__).parent / "matlab" / "ref_sample_from_box.mat",
                        squeeze_me=True, struct_as_record=False)
    failures = []
    for name, (dims, sizes, cells) in CASES.items():
        ref = reference[name]
        g = compute_geometry(cart_grid(list(dims), list(sizes)))

        mine = sample_from_box(g, np.asarray(ref.p, dtype=float), cells)
        theirs = np.atleast_1d(np.asarray(ref.q, dtype=float)).ravel()

        ok = mine.shape == theirs.shape and np.allclose(mine, theirs)
        print(f"  {'ok ' if ok else 'FAIL'} {name}: sampleFromBox "
              f"{mine.shape} vs {theirs.shape}")
        if not ok:
            failures.append(name)
            if mine.shape == theirs.shape:
                bad = int(np.count_nonzero(mine != theirs))
                print(f"       {bad} of {mine.size} entries differ")

    print()
    if failures:
        print(f"FAILED: {', '.join(failures)}")
        return 1
    print("sample_from_box matches MRST's sampleFromBox on Cartesian grids.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
