"""Check `compute_vertical_grid_intersection` against MRST.

Reference: scripts/matlab/ref_vertical_intersection.mat. Regenerate from the
MRST root with:

    matlab -batch "startup; run('.../scripts/matlab/dump_vertical_intersection_reference.m')"

`cellIx` and `cumlength` are compared directly - cell numbering agrees between
the two `cartGrid`s, and `cellIx` is sorted by construction. The vertex arrays
are compared as **sorted point sets**: the polygon ordering comes from an
`atan2` about each cell's own centre, and two points at the same angle can tie
either way, so the ordering is not something to pin. What must match is which
points were found, how many there are, and how they group into polygons.

The `spanning` case is the one that exercises the geometry hardest - the
polyline runs past both ends of the grid, so no cell's polygon is clipped and
every one should come back with a full set of corners.
"""

import sys
from pathlib import Path

import numpy as np
from scipy.io import loadmat

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pymrst.gridprocessing import cart_grid, compute_geometry               # noqa: E402
from pymrst.utils.gridtools.add_bounding_box_fields import (                # noqa: E402
    add_bounding_box_fields,
)
from pymrst.utils.gridtools.compute_vertical_grid_intersection import (     # noqa: E402
    compute_vertical_grid_intersection,
)

CASES = {
    "straight": np.array([[0.5, 0.5], [3.5, 0.5]]),
    "diagonal": np.array([[0.2, 0.2], [3.8, 3.8]]),
    "bend": np.array([[0.5, 0.5], [2.5, 0.5], [2.5, 3.5]]),
    "spanning": np.array([[-1.0, 0.5], [5.0, 0.5]]),
}


def sorted_rows(a, width):
    a = np.asarray(a, dtype=float).reshape(-1, width)
    a = np.round(a, 9)
    return a[np.lexsort(tuple(a[:, d] for d in range(width)))]


def main() -> int:
    reference = loadmat(
        Path(__file__).parent / "matlab" / "ref_vertical_intersection.mat",
        squeeze_me=True, struct_as_record=False)
    g = add_bounding_box_fields(
        compute_geometry(cart_grid([4, 4, 3], [4.0, 4.0, 3.0])))
    failures = []

    for name, segments in CASES.items():
        ref = reference[name]
        out = compute_vertical_grid_intersection(g, segments)

        checks = [
            ("cellIx", np.array_equal(
                np.asarray(out["cellIx"], dtype=np.int64),
                np.asarray(ref.cellIx, dtype=np.int64).ravel() - 1),
             f"{out['cellIx'].size} cells"),
            ("cumlength", np.allclose(out["cumlength"],
                                      np.asarray(ref.cumlength).ravel()), ""),
            ("point count", out["coords3D"].shape[0] == int(ref.nPoints),
             f"{out['coords3D'].shape[0]} vs {int(ref.nPoints)}"),
            ("polygon shape", out["nodes"].shape
             == (int(ref.nPoly), int(ref.maxWidth)),
             f"{out['nodes'].shape} vs {(int(ref.nPoly), int(ref.maxWidth))}"),
            ("coords3D", np.allclose(sorted_rows(out["coords3D"], 3),
                                     sorted_rows(ref.c3sorted, 3),
                                     atol=1e-9), ""),
            ("coords2D", np.allclose(sorted_rows(out["coords2D"], 2),
                                     sorted_rows(ref.c2sorted, 2),
                                     atol=1e-9), ""),
        ]
        for label, ok, detail in checks:
            print(f"  {'ok ' if ok else 'FAIL'} {name}: {label}"
                  f"{'  ' + detail if detail else ''}")
            if not ok:
                failures.append(f"{name} {label}")

    print()
    if failures:
        print(f"FAILED: {', '.join(failures)}")
        return 1
    print("compute_vertical_grid_intersection matches MRST's "
          "computeVerticalGridIntersection.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
