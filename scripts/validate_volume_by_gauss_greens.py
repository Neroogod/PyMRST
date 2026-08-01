"""Check `volume_by_gauss_greens` against MRST (scripts/matlab/ref_gauss_greens.mat).

Regenerate the reference from the MRST root with:

    matlab -batch "startup; run('.../scripts/matlab/dump_gauss_greens_reference.m')"

Comparison is **entry-wise by cell**, which is possible here where it was not
for the bounding boxes: `cartGrid` numbers cells in the same natural order in
both codes, and this function returns one row per cell. The check asserts that
correspondence first, against `cells.volumes`, rather than assuming it.

The twisted case rebuilds MATLAB's grid from the dumped node coordinates, so
both codes see the same warped geometry. It is the case that matters: it is
where the method stops being exact, and a port that quietly differed there
would still pass on Cartesian grids.
"""

import sys
from pathlib import Path

import numpy as np
from scipy.io import loadmat

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pymrst.gridprocessing import cart_grid, compute_geometry           # noqa: E402
from pymrst.utils.gridtools.volume_by_gauss_greens import (             # noqa: E402
    volume_by_gauss_greens,
)


def main() -> int:
    reference = loadmat(Path(__file__).parent / "matlab" / "ref_gauss_greens.mat",
                        squeeze_me=True, struct_as_record=False)
    failures = []

    cases = [("cart", cart_grid([3, 2, 2], [6.0, 4.0, 2.0]), None),
             ("cart2d", cart_grid([3, 3], [3.0, 3.0]), None)]

    twisted = cart_grid([4, 4, 2], [4.0, 4.0, 2.0])
    twisted.nodes.coords = np.asarray(reference["twisted"].coords, dtype=float)
    cases.append(("twisted", twisted, None))

    for name, grid, _ in cases:
        ref = reference[name]
        g = compute_geometry(grid)

        # The cells must line up before comparing anything per cell.
        volumes = np.asarray(g.cells.volumes, dtype=float)
        aligned = np.allclose(volumes, np.asarray(ref.vol, dtype=float).ravel())
        print(f"  {'ok ' if aligned else 'FAIL'} {name}: cell numbering agrees")
        if not aligned:
            failures.append(f"{name}: cell numbering")
            continue

        mine = volume_by_gauss_greens(g)
        theirs = np.asarray(ref.V, dtype=float)
        ok = mine.shape == theirs.shape and np.allclose(mine, theirs, rtol=1e-10)
        worst = (np.max(np.abs(mine - theirs)) if mine.shape == theirs.shape
                 else float("nan"))
        print(f"  {'ok ' if ok else 'FAIL'} {name}: volumeByGaussGreens "
              f"(max diff {worst:.2e})")
        if not ok:
            failures.append(name)

        if name == "twisted":
            # The documented inexactness must show up in both codes alike.
            gap = np.max(np.abs(theirs[:, 0] - np.asarray(ref.vol).ravel()))
            print(f"  ..  MRST itself differs from computeGeometry by "
                  f"{gap:.2e} here, as expected for non-planar faces")
            if gap < 1e-9:
                failures.append("twisted: expected MRST to be inexact too")

    print()
    if failures:
        print(f"FAILED: {', '.join(failures)}")
        return 1
    print("volume_by_gauss_greens matches MRST's volumeByGaussGreens.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
