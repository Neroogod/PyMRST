"""Check `transform_3d_to_2d_grid` against MRST (scripts/matlab/ref_transform_2d.mat).

Regenerate the reference from the MRST root with:

    matlab -batch "startup; run('.../scripts/matlab/dump_transform_2d_reference.m')"

**The reference required reviving MRST's function.** `transform3Dto2Dgrid`
reads `faces.numNodes` and `cells.numFaces`, which MRST grids no longer carry,
so it fails on `computeGeometry(cartGrid(...))` with
`Unrecognized field name "numFaces"`. The dump script supplies the two legacy
fields; without them there is no reference to compare against at all.

Faces are matched by **centroid**, since the two codes number them
differently. Cells are compared position by position: both number the cells of
a `cartGrid` the same way and neither transform reorders them, which the volume
comparison confirms before anything else is checked.

Areas are compared to MRST's *scaled* areas even though the port recomputes
them from the collapsed geometry - agreement between those two is the point,
not an accident to be tolerated.
"""

import sys
from pathlib import Path

import numpy as np
from scipy.io import loadmat

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pymrst.gridprocessing import cart_grid, compute_geometry              # noqa: E402
from pymrst.utils.gridtools.transform_3d_to_2d_grid import (               # noqa: E402
    transform_3d_to_2d_grid,
)

CASES = {"x": ([1, 4, 3], [1.0, 4.0, 3.0]), "y": ([4, 1, 3], [4.0, 1.0, 3.0]),
         "z": ([4, 3, 1], [4.0, 3.0, 1.0]),
         "thick": ([1, 5, 4], [3.0, 5.0, 4.0])}


def by_centroid(centroids, values):
    """Map a rounded centroid to its value, for numbering-free comparison."""
    return {tuple(np.round(c, 9)): v for c, v in zip(centroids, values)}


def main() -> int:
    reference = loadmat(Path(__file__).parent / "matlab" / "ref_transform_2d.mat",
                        squeeze_me=True, struct_as_record=False)
    failures = []
    for name, (dims, sizes) in CASES.items():
        ref = reference[name]
        g = compute_geometry(transform_3d_to_2d_grid(
            compute_geometry(cart_grid(list(dims), list(sizes)))))

        checks = [
            ("counts", (g.cells.num, g.faces.num, g.nodes.num)
             == (int(ref.numCells), int(ref.numFaces), int(ref.numNodes))),
            ("cartDims", tuple(g.cartDims)
             == tuple(np.atleast_1d(np.asarray(ref.cartDims)).astype(int))),
            ("cell volumes", np.allclose(np.asarray(g.cells.volumes),
                                         np.asarray(ref.volumes).ravel())),
            ("cell centroids", np.allclose(np.asarray(g.cells.centroids),
                                           np.asarray(ref.ccent, dtype=float))),
        ]

        mine = by_centroid(np.asarray(g.faces.centroids, float),
                           np.asarray(g.faces.areas, float))
        theirs = by_centroid(np.asarray(ref.fcent, float),
                             np.asarray(ref.fareas, float).ravel())
        checks.append(("face centroids", set(mine) == set(theirs)))
        checks.append(("face areas", set(mine) == set(theirs) and all(
            np.isclose(mine[k], theirs[k]) for k in mine)))

        mine_n = by_centroid(np.asarray(g.faces.centroids, float),
                             [tuple(sorted(p)) for p in
                              np.asarray(g.faces.neighbors, np.int64)])
        theirs_n = by_centroid(np.asarray(ref.fcent, float),
                               [tuple(sorted(p)) for p in
                                np.asarray(ref.fneigh, np.int64) - 1])
        checks.append(("face neighbours", set(mine_n) == set(theirs_n) and all(
            mine_n[k] == theirs_n[k] for k in mine_n)))

        for label, ok in checks:
            print(f"  {'ok ' if ok else 'FAIL'} {name}: {label}")
            if not ok:
                failures.append(f"{name}: {label}")

    print()
    if failures:
        print(f"FAILED: {', '.join(failures)}")
        return 1
    print("transform_3d_to_2d_grid matches MRST's transform3Dto2Dgrid "
          "(with its legacy fields restored).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
